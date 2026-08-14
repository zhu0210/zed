#[cfg(target_os = "linux")]
use crate::wgpu_context::{EXTERNAL_SEMAPHORE_FD_EXTENSION, QUEUE_FAMILY_FOREIGN_EXTENSION};
use crate::{CompositorGpuHint, WgpuAtlas, WgpuContext};
use anyhow::{Context as _, Result};
#[cfg(target_os = "linux")]
use ash::{khr::external_semaphore_fd, vk};
use bytemuck::{Pod, Zeroable};
use gpui::{
    AtlasTextureId, Background, Bounds, DevicePixels, ExternalFrameAcquisition, ExternalFrameOutcome, GpuSpecs, Path, Point, PrimitiveBatch,
    ScaledPixels, Scene, Size, get_gamma_correction_ratios,
};
#[cfg(target_os = "linux")]
use gpui::{ExternalFrameRequest, ExternalNv12Frame, ExternalOwnership};
use log::warn;
#[cfg(not(target_family = "wasm"))]
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::cell::RefCell;
use std::num::NonZeroU64;
use std::ops::Range;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

const MAX_INSTANCE_BUFFER_SIZE: u64 = 256 * 1024 * 1024;

const INSTANCE_TEXTURE_TEXEL_SIZE: u64 = 16;

/// Shader variant for backends with storage buffer support: the shared shader
/// logic plus the storage-buffer instance transport.
const STORAGE_BUFFER_SHADERS: &str = concat!(
    include_str!("shaders.wgsl"),
    include_str!("shaders_storage.wgsl"),
);

/// Shader variant for WebGL2, which has no storage buffers: the shared shader
/// logic plus the texture-based instance transport.
const WEBGL_SHADERS: &str = concat!(
    include_str!("shaders.wgsl"),
    include_str!("shaders_webgl.wgsl"),
);

/// Subpixel text rendering requires dual-source blending, which WebGL2 lacks, so
/// this variant only ever runs with the storage-buffer transport. The `enable`
/// directive must precede all declarations.
const SUBPIXEL_SHADERS: &str = concat!(
    "enable dual_source_blending;\n",
    include_str!("shaders.wgsl"),
    include_str!("shaders_storage.wgsl"),
    include_str!("shaders_subpixel.wgsl"),
);

fn least_common_multiple(left: u64, right: u64) -> u64 {
    let mut first = left;
    let mut second = right;
    while second != 0 {
        let remainder = first % second;
        first = second;
        second = remainder;
    }
    left / first * right
}

#[cfg(target_os = "linux")]
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};


#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalParams {
    viewport_size: [f32; 2],
    premultiplied_alpha: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PodBounds {
    origin: [f32; 2],
    size: [f32; 2],
}

impl From<Bounds<ScaledPixels>> for PodBounds {
    fn from(bounds: Bounds<ScaledPixels>) -> Self {
        Self {
            origin: [bounds.origin.x.0, bounds.origin.y.0],
            size: [bounds.size.width.0, bounds.size.height.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SurfaceParams {
    bounds: PodBounds,
    content_mask: PodBounds,
    yuv_to_rgb: [[f32; 4]; 4],
}

impl SurfaceParams {
    fn new(
        bounds: PodBounds,
        content_mask: PodBounds,
        transform: gpui::Nv12ColorTransform,
    ) -> Self {
        Self {
            bounds,
            content_mask,
            yuv_to_rgb: transform.yuv_to_rgb,
        }
    }
}

fn nv12_plane_view_descriptor(
    aspect: wgpu::TextureAspect,
) -> Option<wgpu::TextureViewDescriptor<'static>> {
    let (format, label) = match aspect {
        wgpu::TextureAspect::Plane0 => (wgpu::TextureFormat::R8Unorm, "nv12_plane_0"),
        wgpu::TextureAspect::Plane1 => (wgpu::TextureFormat::Rg8Unorm, "nv12_plane_1"),
        _ => return None,
    };
    Some(wgpu::TextureViewDescriptor {
        label: Some(label),
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        aspect,
        ..Default::default()
    })
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SurfaceInstance {
    bounds: PodBounds,
    content_mask: PodBounds,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GammaParams {
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    is_bgr: u32,
    _pad: u32,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct PathSprite {
    bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct PathRasterizationVertex {
    xy_position: Point<ScaledPixels>,
    st_position: Point<f32>,
    color: Background,
    bounds: Bounds<ScaledPixels>,
}

pub struct WgpuSurfaceConfig {
    pub size: Size<DevicePixels>,
    pub transparent: bool,
    /// Preferred presentation mode. When `Some`, the renderer will use this
    /// mode if supported by the surface, falling back to `Fifo`.
    /// When `None`, defaults to `Fifo` (VSync).
    ///
    /// Mobile platforms may prefer `Mailbox` (triple-buffering) to avoid
    /// blocking in `get_current_texture()` during lifecycle transitions.
    pub preferred_present_mode: Option<wgpu::PresentMode>,
}

#[cfg(target_os = "linux")]
type ExternalFrameLease = Box<dyn FnOnce() + Send + Sync + 'static>;

#[cfg(target_os = "linux")]
fn queue_family(ownership: ExternalOwnership) -> u32 {
    match ownership {
        ExternalOwnership::External => vk::QUEUE_FAMILY_EXTERNAL,
        ExternalOwnership::Foreign => vk::QUEUE_FAMILY_FOREIGN_EXT,
    }
}

#[cfg(target_os = "linux")]
/// A producer-owned Vulkan image and its renderer-boundary acquire fence.
pub struct VulkanExternalFrame {
    image: vk::Image,
    size: wgpu::Extent3d,
    format: wgpu::TextureFormat,
    initial_state: wgpu::TextureUses,
    ownership: ExternalOwnership,
    sync_file: Option<OwnedFd>,
    lease: Option<ExternalFrameLease>,
}

#[cfg(target_os = "linux")]
impl VulkanExternalFrame {
    /// Describe an initialized producer image and take ownership of its sync
    /// file descriptor until the renderer consumes it.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following:
    ///
    /// - `image` was created on the Vulkan device used by
    ///   [`PreparedExternalFrame::from_vulkan`].
    /// - The image is a 2D image with one mip level, one array layer, one sample,
    ///   and a format and extent matching `format` and `size`.
    /// - The image was created with `VK_IMAGE_USAGE_SAMPLED_BIT`, matching the
    ///   imported wgpu texture's `TEXTURE_BINDING` usage.
    /// - Its actual Vulkan layout is `GENERAL`, it is currently owned by
    ///   `ownership`, and `initial_state` is the concrete wgpu-tracked
    ///   consumer state established after the acquire transition.
    /// - The image and its backing memory remain valid until the lease callback
    ///   runs.
    /// - The producer completed its release barrier and signaled `sync_file`
    ///   before the frame is consumed.
    pub unsafe fn new(
        image: vk::Image,
        size: wgpu::Extent3d,
        format: wgpu::TextureFormat,
        initial_state: wgpu::TextureUses,
        ownership: ExternalOwnership,
        sync_file: OwnedFd,
        lease: impl FnOnce() + Send + Sync + 'static,
    ) -> anyhow::Result<Self> {
        let lease: ExternalFrameLease = Box::new(lease);
        if let Err(error) = ensure_external_frame_initial_state(initial_state) {
            lease();
            return Err(error);
        }
        Ok(Self {
            image,
            size,
            format,
            initial_state,
            ownership,
            sync_file: Some(sync_file),
            lease: Some(lease),
        })
    }
}

#[cfg(target_os = "linux")]
impl Drop for VulkanExternalFrame {
    fn drop(&mut self) {
        if let Some(lease) = self.lease.take() {
            lease();
        }
    }
}

/// A frame prepared for submission by the renderer.
///
/// Vulkan imports retain their acquire fence until the next draw submission
/// and release their producer lease from the HAL texture drop callback.
pub struct PreparedExternalFrame {
    #[cfg(target_os = "linux")]
    // Drop the Vulkan sync object before the texture's device reference.
    external_sync: Option<VulkanExternalSync>,
    #[cfg(target_os = "linux")]
    initial_state: wgpu::TextureUses,
    #[cfg(target_os = "linux")]
    ownership: Option<ExternalOwnership>,
    texture: Arc<wgpu::Texture>,
}

impl PreparedExternalFrame {
    /// Construct a frame from a texture created for the renderer's device.
    pub fn new(texture: Arc<wgpu::Texture>) -> Self {
        Self {
            #[cfg(target_os = "linux")]
            external_sync: None,
            #[cfg(target_os = "linux")]
            initial_state: wgpu::TextureUses::RESOURCE,
            #[cfg(target_os = "linux")]
            ownership: None,
            texture,
        }
    }

    /// Return the texture sampled for this frame.
    pub fn texture(&self) -> &Arc<wgpu::Texture> {
        &self.texture
    }

    #[cfg(target_os = "linux")]
    /// Import a producer-owned Vulkan image into the renderer's wgpu device.
    ///
    /// The sync file remains pending until the next normal renderer submit;
    /// this lets the draw call stage the wait in wgpu's queue ordering.
    pub fn from_vulkan(
        context: &gpui::GpuContextHandle,
        mut frame: VulkanExternalFrame,
    ) -> anyhow::Result<Self> {
        ensure_external_frame_initial_state(frame.initial_state)?;
        anyhow::ensure!(
            frame.image != vk::Image::null(),
            "external image handle is null"
        );
        anyhow::ensure!(
            frame.size.width != 0
                && frame.size.height != 0
                && frame.size.depth_or_array_layers == 1,
            "external frame must be a non-empty 2D image"
        );
        if context.adapter.get_info().backend != wgpu::Backend::Vulkan {
            anyhow::bail!("Vulkan external frames require a Vulkan adapter");
        }

        // SAFETY: the adapter and device come from the same wgpu context. The
        // producer owns the image and promises that its descriptor and initial
        // Vulkan layout match these values until wgpu releases the texture.
        let hal_device = unsafe { context.device.as_hal::<wgpu::hal::api::Vulkan>() }
            .ok_or_else(|| anyhow::anyhow!("Vulkan HAL device is unavailable"))?;
        if !hal_device
            .enabled_device_extensions()
            .contains(&EXTERNAL_SEMAPHORE_FD_EXTENSION)
        {
            anyhow::bail!("Vulkan external semaphore fd extension is not enabled");
        }
        if frame.ownership == ExternalOwnership::Foreign
            && !hal_device
                .enabled_device_extensions()
                .contains(&QUEUE_FAMILY_FOREIGN_EXTENSION)
        {
            anyhow::bail!("Vulkan queue-family-foreign extension is not enabled");
        }

        let raw_device = hal_device.raw_device().clone();
        let external_semaphore_fd = external_semaphore_fd::Device::new(
            hal_device.shared_instance().raw_instance(),
            &raw_device,
        );
        let sync_file = frame
            .sync_file
            .take()
            .ok_or_else(|| anyhow::anyhow!("external frame sync file was already consumed"))?;
        let external_sync = VulkanExternalSync {
            fd: Some(sync_file),
            device: raw_device,
            external_semaphore_fd,
            semaphore: None,
            #[cfg(test)]
            completion_probe: None,
        };

        let texture_descriptor = wgpu::TextureDescriptor {
            label: Some("external_frame"),
            size: frame.size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: frame.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let hal_texture_descriptor = wgpu::hal::TextureDescriptor {
            label: Some("external_frame"),
            size: frame.size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: frame.format,
            usage: wgpu::TextureUses::RESOURCE,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: Vec::new(),
        };
        let lease = frame.lease.take();
        // SAFETY: The producer keeps `frame.image` valid, and the HAL descriptor matches the
        // image while the drop callback retains the producer lease.
        let hal_texture = unsafe {
            hal_device.texture_from_raw(
                frame.image,
                &hal_texture_descriptor,
                Some(Box::new(move || {
                    if let Some(lease) = lease {
                        lease();
                    }
                })),
                wgpu::hal::vulkan::TextureMemory::External,
            )
        };
        // SAFETY: `hal_texture` came from this device and `frame.initial_state` is the concrete
        // wgpu-tracked consumer state established after the external GENERAL-layout acquire.
        let texture = unsafe {
            context
                .device
                .create_texture_from_hal::<wgpu::hal::api::Vulkan>(
                    hal_texture,
                    &texture_descriptor,
                    frame.initial_state,
                )
        };

        Ok(Self {
            external_sync: Some(external_sync),
            initial_state: frame.initial_state,
            ownership: Some(frame.ownership),
            texture: Arc::new(texture),
        })
    }

    #[cfg(target_os = "linux")]
    fn from_external_nv12(
        context: &gpui::GpuContextHandle,
        frame: ExternalNv12Frame,
    ) -> Result<Self, ExternalFrameOutcome> {
        if context.adapter.get_info().backend != wgpu::Backend::Vulkan
            || !context
                .device
                .features()
                .contains(wgpu::Features::TEXTURE_FORMAT_NV12)
        {
            return Err(ExternalFrameOutcome::Unsupported);
        }

        // SAFETY: the caller constructed the frame for this renderer device and
        // promised that the texture is an initialized Vulkan NV12 image.
        let Some(hal_device) = (unsafe { context.device.as_hal::<wgpu::hal::api::Vulkan>() })
        else {
            return Err(ExternalFrameOutcome::Unsupported);
        };
        if !hal_device
            .enabled_device_extensions()
            .contains(&EXTERNAL_SEMAPHORE_FD_EXTENSION)
        {
            return Err(ExternalFrameOutcome::Unsupported);
        }

        let (texture, sync_file, ownership) = frame.into_parts();
        if ownership == ExternalOwnership::Foreign
            && !hal_device
                .enabled_device_extensions()
                .contains(&QUEUE_FAMILY_FOREIGN_EXTENSION)
        {
            return Err(ExternalFrameOutcome::Unsupported);
        }

        let raw_device = hal_device.raw_device().clone();
        let external_semaphore_fd = external_semaphore_fd::Device::new(
            hal_device.shared_instance().raw_instance(),
            &raw_device,
        );
        let external_sync = VulkanExternalSync {
            fd: Some(sync_file),
            device: raw_device,
            external_semaphore_fd,
            semaphore: None,
            #[cfg(test)]
            completion_probe: None,
        };

        Ok(Self {
            external_sync: Some(external_sync),
            initial_state: wgpu::TextureUses::RESOURCE,
            ownership: Some(ownership),
            texture,
        })
    }

    #[cfg(target_os = "linux")]
    fn is_external(&self) -> bool {
        self.ownership.is_some()
    }

    #[cfg(target_os = "linux")]
    fn import_sync(&mut self, queue: &wgpu::Queue) -> Result<(), vk::Result> {
        let Some(external_sync) = self.external_sync.as_mut() else {
            return Ok(());
        };

        // SAFETY: The queue belongs to the same Vulkan device as the imported semaphore.
        let Some(queue_hal) = (unsafe { queue.as_hal::<wgpu::hal::api::Vulkan>() }) else {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        };
        external_sync.import(&queue_hal)
    }

    #[cfg(target_os = "linux")]
    fn encode_acquire_ownership(&self, encoder: &mut wgpu::CommandEncoder) -> Result<(), ()> {
        let Some(ownership) = self.ownership else {
            return Ok(());
        };
        let Some(texture) = (unsafe { self.texture.as_hal::<wgpu::hal::api::Vulkan>() }) else {
            return Err(());
        };
        let range = wgpu::ImageSubresourceRange {
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        };
        let mut acquired = false;
        // SAFETY: `texture` and `encoder` are from the same Vulkan wgpu device; the encoder is
        // recording outside a render pass, and the descriptor records the producer's exact
        // externally owned GENERAL layout into the concrete tracked consumer state and records
        // the external queue-family ownership.
        unsafe {
            encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|hal_encoder| {
                let Some(hal_encoder) = hal_encoder else {
                    return;
                };
                hal_encoder.acquire_external_texture_ownership(
                    &texture,
                    &range,
                    self.initial_state,
                    queue_family(ownership),
                );
                acquired = true;
            });
        }
        acquired.then_some(()).ok_or(())
    }

    #[cfg(target_os = "linux")]
    fn encode_release_ownership(&self, encoder: &mut wgpu::CommandEncoder) -> Result<(), ()> {
        let Some(ownership) = self.ownership else {
            return Ok(());
        };
        let Some(texture) = (unsafe { self.texture.as_hal::<wgpu::hal::api::Vulkan>() }) else {
            return Err(());
        };
        let range = wgpu::ImageSubresourceRange {
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        };
        let mut released = false;
        // SAFETY: `texture` and `encoder` are from the same Vulkan wgpu device; the encoder is
        // recording outside a render pass and RESOURCE is the tracked state after sampling.
        unsafe {
            encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|hal_encoder| {
                let Some(hal_encoder) = hal_encoder else {
                    return;
                };
                hal_encoder.release_external_texture_ownership(
                    &texture,
                    &range,
                    queue_family(ownership),
                );
                released = true;
            });
        }
        released.then_some(()).ok_or(())
    }

    #[cfg(target_os = "linux")]
    fn on_submitted(&mut self, queue: &wgpu::Queue) {
        if !self.is_external() {
            return;
        }
        let external_sync = self.external_sync.take();
        let texture = Arc::clone(&self.texture);
        queue.on_submitted_work_done(move || {
            drop(external_sync);
            drop(texture);
        });
    }
}

enum PendingExternalFrame {
    Prepared(PreparedExternalFrame),
    #[cfg(target_os = "linux")]
    Nv12(ExternalNv12Frame),
}

#[cfg(target_os = "linux")]
impl ExternalFrameState<PendingExternalFrame> {
    fn selection_for_scene(&self, scene: &Scene) -> Option<ExternalFrameSlot> {
        self.select_matching(|frame| match frame {
            PendingExternalFrame::Prepared(frame) => scene_contains_texture(scene, frame.texture()),
            PendingExternalFrame::Nv12(_) => false,
        })
    }

    fn selected(&self, slot: ExternalFrameSlot) -> Option<&PendingExternalFrame> {
        match slot {
            ExternalFrameSlot::Latest => self.latest.as_ref(),
            ExternalFrameSlot::Displayed => self.displayed.as_ref(),
        }
    }

    fn selected_mut(&mut self, slot: ExternalFrameSlot) -> Option<&mut PendingExternalFrame> {
        match slot {
            ExternalFrameSlot::Latest => self.latest.as_mut(),
            ExternalFrameSlot::Displayed => self.displayed.as_mut(),
        }
    }

    fn selected_is_external(&self, slot: ExternalFrameSlot) -> bool {
        matches!(self.selected(slot), Some(PendingExternalFrame::Prepared(frame)) if frame.is_external())
    }

    fn clear_selected_latest(&mut self, slot: ExternalFrameSlot) {
        if slot == ExternalFrameSlot::Latest {
            self.latest = None;
        }
    }

    fn prepare_frame(&mut self, context: Option<&gpui::GpuContextHandle>) -> ExternalFrameOutcome {
        let Some(frame) = self.latest.take() else {
            return ExternalFrameOutcome::Accepted;
        };
        let frame = match frame {
            PendingExternalFrame::Prepared(frame) => frame,
            PendingExternalFrame::Nv12(frame) => {
                let Some(context) = context else {
                    return ExternalFrameOutcome::Unsupported;
                };
                match PreparedExternalFrame::from_external_nv12(context, frame) {
                    Ok(frame) => frame,
                    Err(outcome) => return outcome,
                }
            }
        };

        self.latest = Some(PendingExternalFrame::Prepared(frame));
        ExternalFrameOutcome::Accepted
    }

    fn import_sync(
        &mut self,
        slot: ExternalFrameSlot,
        queue: &wgpu::Queue,
    ) -> ExternalFrameOutcome {
        if slot == ExternalFrameSlot::Displayed {
            return ExternalFrameOutcome::Accepted;
        }
        let Some(frame) = self.selected_mut(slot) else {
            return ExternalFrameOutcome::Accepted;
        };
        let PendingExternalFrame::Prepared(frame) = frame else {
            self.clear_selected_latest(slot);
            return ExternalFrameOutcome::FatalFailure;
        };
        if let Err(error) = frame.import_sync(queue) {
            self.clear_selected_latest(slot);
            return classify_external_sync_error(error);
        }
        ExternalFrameOutcome::Accepted
    }

    fn encode_acquire(
        &mut self,
        slot: ExternalFrameSlot,
        encoder: &mut wgpu::CommandEncoder,
    ) -> ExternalFrameOutcome {
        let Some(frame) = self.selected(slot) else {
            return ExternalFrameOutcome::Accepted;
        };
        let frame = match frame {
            PendingExternalFrame::Prepared(frame) => frame,
            PendingExternalFrame::Nv12(_) => return ExternalFrameOutcome::Accepted,
        };
        if frame.encode_acquire_ownership(encoder).is_err() {
            self.clear_selected_latest(slot);
            return ExternalFrameOutcome::FatalFailure;
        }
        ExternalFrameOutcome::Accepted
    }

    fn encode_release(
        &mut self,
        slot: ExternalFrameSlot,
        encoder: &mut wgpu::CommandEncoder,
    ) -> ExternalFrameOutcome {
        let Some(frame) = self.selected(slot) else {
            return ExternalFrameOutcome::Accepted;
        };
        let frame = match frame {
            PendingExternalFrame::Prepared(frame) => frame,
            PendingExternalFrame::Nv12(_) => return ExternalFrameOutcome::FatalFailure,
        };
        if frame.encode_release_ownership(encoder).is_err() {
            self.clear_selected_latest(slot);
            return ExternalFrameOutcome::FatalFailure;
        }
        ExternalFrameOutcome::Accepted
    }

    fn commit_after_submission_with_queue(
        &mut self,
        slot: Option<ExternalFrameSlot>,
        queue: &wgpu::Queue,
    ) {
        match slot {
            Some(ExternalFrameSlot::Latest) => {
                if let Some(PendingExternalFrame::Prepared(mut frame)) = self.latest.take() {
                    frame.on_submitted(queue);
                    self.displayed = Some(PendingExternalFrame::Prepared(frame));
                }
            }
            Some(ExternalFrameSlot::Displayed) => {
                if let Some(PendingExternalFrame::Prepared(frame)) = self.displayed.as_mut() {
                    frame.on_submitted(queue);
                }
            }
            None => {}
        }
    }
}

#[cfg(target_os = "linux")]
fn scene_contains_texture(scene: &Scene, texture: &Arc<wgpu::Texture>) -> bool {
    scene.surfaces.iter().any(|surface| match &surface.content {
        gpui::SurfaceContent::WgpuTexture(candidate) => Arc::ptr_eq(candidate, texture),
        gpui::SurfaceContent::WgpuTextureNv12Multiplanar {
            texture: candidate, ..
        } => Arc::ptr_eq(candidate, texture),
        #[allow(unreachable_patterns)]
        _ => false,
    })
}

#[cfg(target_os = "linux")]
fn ensure_external_frame_initial_state(initial_state: wgpu::TextureUses) -> anyhow::Result<()> {
    anyhow::ensure!(
        !initial_state.contains(wgpu::TextureUses::UNINITIALIZED),
        "external frame initial state must describe initialized producer contents"
    );
    anyhow::ensure!(
        !initial_state.is_empty()
            && !initial_state.intersects(
                wgpu::TextureUses::COMPLEX
                    | wgpu::TextureUses::UNKNOWN
                    | wgpu::TextureUses::TRANSIENT,
            ),
        "external frame initial state must be a concrete texture usage"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn classify_external_sync_error(error: vk::Result) -> ExternalFrameOutcome {
    match error {
        vk::Result::ERROR_INITIALIZATION_FAILED
        | vk::Result::ERROR_DEVICE_LOST
        | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
        | vk::Result::ERROR_OUT_OF_HOST_MEMORY => ExternalFrameOutcome::FatalFailure,
        _ => ExternalFrameOutcome::TransientFailure,
    }
}

#[cfg(target_os = "linux")]
struct VulkanExternalSync {
    fd: Option<OwnedFd>,
    device: ash::Device,
    external_semaphore_fd: external_semaphore_fd::Device,
    semaphore: Option<vk::Semaphore>,
    #[cfg(test)]
    completion_probe: Option<Arc<std::sync::atomic::AtomicBool>>,
}

#[cfg(target_os = "linux")]
impl VulkanExternalSync {
    fn sync_file_import_info(
        semaphore: vk::Semaphore,
        fd: i32,
    ) -> vk::ImportSemaphoreFdInfoKHR<'static> {
        vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(semaphore)
            .flags(vk::SemaphoreImportFlags::TEMPORARY)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
            .fd(fd)
    }

    fn import(&mut self, queue: &wgpu::hal::vulkan::Queue) -> Result<(), vk::Result> {
        if self.semaphore.is_some() {
            return Ok(());
        }
        let fd = self
            .fd
            .take()
            .ok_or(vk::Result::ERROR_INVALID_EXTERNAL_HANDLE)?;
        // SAFETY: `self.device` is the live Vulkan device associated with the HAL queue.
        let semaphore = unsafe {
            self.device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
        };
        let fd = fd.into_raw_fd();
        let import_info = Self::sync_file_import_info(semaphore, fd);
        // SAFETY: The extension function and semaphore were obtained from `self.device`, and
        // ownership of the valid raw fd is transferred to Vulkan by this import.
        let import_result = unsafe { self.external_semaphore_fd.import_semaphore_fd(&import_info) };
        if let Err(error) = import_result {
            // Vulkan retains ownership of the fd only after a successful
            // import. Reconstitute it on failure before destroying the empty
            // semaphore.
            // SAFETY: Import failed, so Vulkan did not take ownership of `fd`; `semaphore` was
            // created above and has not been submitted.
            unsafe {
                drop(OwnedFd::from_raw_fd(fd));
                self.device.destroy_semaphore(semaphore, None);
            }
            return Err(error);
        }

        queue.add_wait_semaphore(semaphore, None, vk::PipelineStageFlags::TOP_OF_PIPE);
        self.semaphore = Some(semaphore);
        Ok(())
    }

    fn destroy_semaphore(&mut self) {
        if let Some(semaphore) = self.semaphore.take() {
            // SAFETY: The semaphore is either not submitted yet or this method is called from
            // the queue completion callback after its wait has finished.
            unsafe {
                self.device.destroy_semaphore(semaphore, None);
            }
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for VulkanExternalSync {
    fn drop(&mut self) {
        self.destroy_semaphore();
        #[cfg(test)]
        if let Some(probe) = self.completion_probe.take() {
            probe.store(true, std::sync::atomic::Ordering::Release);
        }
    }
}

/// Input to the external-frame seam.
pub enum ExternalFrame {
    /// A prepared texture to make the latest pending frame.
    Prepared(PreparedExternalFrame),
    /// The producer could not acquire a frame this tick.
    TransientFailure,
    /// The producer encountered a failure that requires intervention.
    FatalFailure,
}

struct ExternalFrameState<T> {
    latest: Option<T>,
    displayed: Option<T>,
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExternalFrameSlot {
    Latest,
    Displayed,
}

impl<T> Default for ExternalFrameState<T> {
    fn default() -> Self {
        Self {
            latest: None,
            displayed: None,
        }
    }
}

impl<T> ExternalFrameState<T> {
    fn clear(&mut self) {
        self.latest = None;
        self.displayed = None;
    }

    fn stage(
        &mut self,
        backend: wgpu::Backend,
        acquisition: ExternalFrameAcquisition,
        frame: Option<T>,
    ) -> ExternalFrameOutcome {
        let outcome = ExternalFrameOutcome::classify(backend, acquisition);
        match outcome {
            ExternalFrameOutcome::Accepted => {
                if let Some(frame) = frame {
                    self.latest = Some(frame);
                }
            }
            ExternalFrameOutcome::TransientFailure | ExternalFrameOutcome::FatalFailure => {
                self.latest = None;
            }
            ExternalFrameOutcome::Unsupported => {}
        }
        outcome
    }

    #[cfg(target_os = "linux")]
    fn select_matching(&self, matches: impl Fn(&T) -> bool) -> Option<ExternalFrameSlot> {
        if let Some(frame) = self.latest.as_ref() {
            if matches(frame) {
                return Some(ExternalFrameSlot::Latest);
            }
        }
        if let Some(frame) = self.displayed.as_ref() {
            if matches(frame) {
                return Some(ExternalFrameSlot::Displayed);
            }
        }
        None
    }

    #[cfg(any(test, not(target_os = "linux")))]
    fn commit_after_submission(&mut self) {
        if let Some(frame) = self.latest.take() {
            self.displayed = Some(frame);
        }
    }

    #[cfg(test)]
    fn latest(&self) -> Option<&T> {
        self.latest.as_ref()
    }

    fn displayed(&self) -> Option<&T> {
        self.displayed.as_ref()
    }
}

struct WgpuPipelines {
    quads: wgpu::RenderPipeline,
    shadows: wgpu::RenderPipeline,
    path_rasterization: wgpu::RenderPipeline,
    paths: wgpu::RenderPipeline,
    underlines: wgpu::RenderPipeline,
    mono_sprites: wgpu::RenderPipeline,
    subpixel_sprites: Option<wgpu::RenderPipeline>,
    poly_sprites: wgpu::RenderPipeline,
    #[allow(dead_code)]
    surfaces: wgpu::RenderPipeline,
    surface_rgba: wgpu::RenderPipeline,
}

/// One frame allocation of instance data, ready to bind.
struct InstanceBinding {
    bind_group: wgpu::BindGroup,
    /// Index of the allocation's first instance within the bound data. Always
    /// zero on the storage-buffer path, where the binding offset already
    /// positions the array; on the WebGL texture path the shader indexes the
    /// shared instance texture absolutely, so draws must offset their
    /// instance (or vertex) ranges by this value.
    first_instance: u32,
}

struct InstanceBindings {
    quads: InstanceBinding,
    shadows: InstanceBinding,
    underlines: InstanceBinding,
    monochrome_sprites: InstanceBinding,
    subpixel_sprites: InstanceBinding,
    polychrome_sprites: InstanceBinding,
}

struct WgpuBindGroupLayouts {
    globals: wgpu::BindGroupLayout,
    instances: wgpu::BindGroupLayout,
    texture: wgpu::BindGroupLayout,
    surfaces: wgpu::BindGroupLayout,
}

/// Shared GPU context reference, used to coordinate device recovery across multiple windows.
pub type GpuContext = Rc<RefCell<Option<WgpuContext>>>;

enum InstanceData {
    Storage(wgpu::Buffer),
    // WebGL2 has no storage buffers. A uint texture keeps the records available to both shader
    // stages while preserving integer and floating-point bit patterns exactly.
    Texture {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        width: u32,
        height: u32,
    },
}

/// GPU resources that must be dropped together during device recovery.
struct WgpuResources {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    pipelines: WgpuPipelines,
    bind_group_layouts: WgpuBindGroupLayouts,
    atlas_sampler: wgpu::Sampler,
    globals_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    path_globals_bind_group: wgpu::BindGroup,
    instance_data: InstanceData,
    path_intermediate_texture: Option<wgpu::Texture>,
    path_intermediate_view: Option<wgpu::TextureView>,
    path_msaa_texture: Option<wgpu::Texture>,
    path_msaa_view: Option<wgpu::TextureView>,
}

impl WgpuResources {
    fn invalidate_intermediate_textures(&mut self) {
        self.path_intermediate_texture = None;
        self.path_intermediate_view = None;
        self.path_msaa_texture = None;
        self.path_msaa_view = None;
    }
}

pub struct WgpuRenderer {
    /// Shared GPU context for device recovery coordination (unused on WASM).
    #[allow(dead_code)]
    context: Option<GpuContext>,
    /// Compositor GPU hint for adapter selection (unused on WASM).
    #[allow(dead_code)]
    compositor_gpu: Option<CompositorGpuHint>,
    resources: Option<WgpuResources>,
    surface_config: wgpu::SurfaceConfiguration,
    atlas: Arc<WgpuAtlas>,
    path_globals_offset: u64,
    gamma_offset: u64,
    instance_data_capacity: u64,
    max_instance_data_size: u64,
    instance_data_alignment: u64,
    uses_webgl_instance_data: bool,
    rendering_params: RenderingParameters,
    is_bgr: bool,
    dual_source_blending: bool,
    adapter_info: wgpu::AdapterInfo,
    transparent_alpha_mode: wgpu::CompositeAlphaMode,
    opaque_alpha_mode: wgpu::CompositeAlphaMode,
    max_texture_size: u32,
    last_error: Arc<Mutex<Option<String>>>,
    failed_frame_count: u32,
    device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    surface_configured: bool,
    needs_redraw: bool,
    external_frames: ExternalFrameState<PendingExternalFrame>,
    last_external_frame_outcome: Option<ExternalFrameOutcome>,
}

impl WgpuRenderer {
    fn resources(&self) -> &WgpuResources {
        self.resources
            .as_ref()
            .expect("GPU resources not available")
    }

    fn resources_mut(&mut self) -> &mut WgpuResources {
        self.resources
            .as_mut()
            .expect("GPU resources not available")
    }

    /// Returns a handle to the GPU resources backing this renderer.
    ///
    /// Returns `None` if the GPU context has not been initialized yet
    /// (before the first frame).
    pub fn gpu_context_handle(&self) -> Option<gpui::GpuContextHandle> {
        let gpu_ctx = self.context.as_ref()?;
        let ctx = gpu_ctx.borrow();
        let wgpu = ctx.as_ref()?;
        Some(gpui::GpuContextHandle {
            device: wgpu.device.clone(),
            queue: wgpu.queue.clone(),
            instance: wgpu.instance.clone(),
            adapter: wgpu.adapter.clone(),
            color_texture_format: wgpu.color_texture_format(),
            supports_dual_source_blending: wgpu.supports_dual_source_blending(),
        })
    }

    /// Stage the latest prepared external frame or report its acquisition
    /// failure. The displayed frame changes only when normal drawing reaches
    /// a successful queue submission. Only Vulkan renderers accept this seam;
    /// other backends return [`ExternalFrameOutcome::Unsupported`] without
    /// changing normal drawing.
    pub fn submit_external_frame(&mut self, frame: ExternalFrame) -> ExternalFrameOutcome {
        let (acquisition, frame) = match frame {
            ExternalFrame::Prepared(frame) => (
                ExternalFrameAcquisition::Prepared,
                Some(PendingExternalFrame::Prepared(frame)),
            ),
            ExternalFrame::TransientFailure => (ExternalFrameAcquisition::TransientFailure, None),
            ExternalFrame::FatalFailure => (ExternalFrameAcquisition::FatalFailure, None),
        };
        let outcome = self
            .external_frames
            .stage(self.adapter_info.backend, acquisition, frame);
        self.last_external_frame_outcome = Some(outcome);
        outcome
    }

    /// Stage a typed external NV12 request for the next normal submission.
    #[cfg(target_os = "linux")]
    pub fn submit_external_frame_request(
        &mut self,
        request: ExternalFrameRequest,
    ) -> ExternalFrameOutcome {
        let (acquisition, frame) = match request {
            ExternalFrameRequest::Prepared(frame) => (
                ExternalFrameAcquisition::Prepared,
                Some(PendingExternalFrame::Nv12(frame)),
            ),
            ExternalFrameRequest::TransientFailure => {
                (ExternalFrameAcquisition::TransientFailure, None)
            }
            ExternalFrameRequest::FatalFailure => (ExternalFrameAcquisition::FatalFailure, None),
        };
        let outcome = self
            .external_frames
            .stage(self.adapter_info.backend, acquisition, frame);
        self.last_external_frame_outcome = Some(outcome);
        outcome
    }

    /// Clear pending and displayed external frames without submitting GPU work.
    ///
    /// Successful external draws already released the image to GENERAL and its
    /// external queue family; their completion callback retains the texture and
    /// producer lease. A never-submitted latest frame remains producer-owned, so
    /// dropping it needs no barrier or submission.
    #[cfg(target_os = "linux")]
    pub fn clear_external_frame(&mut self) -> ExternalFrameOutcome {
        self.external_frames.clear();
        self.last_external_frame_outcome = Some(ExternalFrameOutcome::Accepted);
        ExternalFrameOutcome::Accepted
    }

    /// Return the texture from the last accepted and presented external frame.
    pub fn displayed_external_frame(&self) -> Option<&PreparedExternalFrame> {
        match self.external_frames.displayed()? {
            PendingExternalFrame::Prepared(frame) => Some(frame),
            #[cfg(target_os = "linux")]
            PendingExternalFrame::Nv12(_) => None,
        }
    }

    /// Take the outcome from the most recent external-frame submission or
    /// renderer-boundary synchronization attempt.
    pub fn take_external_frame_outcome(&mut self) -> Option<ExternalFrameOutcome> {
        self.last_external_frame_outcome.take()
    }

    /// Creates a new WgpuRenderer from raw window handles.
    ///
    /// The `gpu_context` is a shared reference that coordinates GPU context across
    /// multiple windows. The first window to create a renderer will initialize the
    /// context; subsequent windows will share it.
    ///
    /// # Safety
    /// The caller must ensure that the window handle remains valid for the lifetime
    /// of the returned renderer.
    #[cfg(not(target_family = "wasm"))]
    pub fn new<W>(
        gpu_context: GpuContext,
        window: &W,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
    ) -> anyhow::Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
    {
        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;

        let target = wgpu::SurfaceTargetUnsafe::RawHandle {
            // Fall back to the display handle already provided via InstanceDescriptor::display.
            raw_display_handle: None,
            raw_window_handle: window_handle.as_raw(),
        };

        // Use the existing context's instance if available, otherwise create a new one.
        // The surface must be created with the same instance that will be used for
        // adapter selection, otherwise wgpu will panic.
        let instance = gpu_context
            .borrow()
            .as_ref()
            .map(|ctx| ctx.instance.clone())
            .unwrap_or_else(|| WgpuContext::instance(Box::new(window.clone())));

        // Safety: The caller guarantees that the window handle is valid for the
        // lifetime of this renderer. In practice, the RawWindow struct is created
        // from the native window handles and the surface is dropped before the window.
        let surface = unsafe {
            instance
                .create_surface_unsafe(target)
                .map_err(|e| anyhow::anyhow!("Failed to create surface: {e}"))?
        };

        let mut ctx_ref = gpu_context.borrow_mut();
        let context = match ctx_ref.as_mut() {
            Some(context) => {
                context.check_compatible_with_surface(&surface)?;
                context
            }
            None => ctx_ref.insert(WgpuContext::new(instance, &surface, compositor_gpu)?),
        };

        let atlas = Arc::new(WgpuAtlas::from_context(context));

        Self::new_internal(
            Some(Rc::clone(&gpu_context)),
            context,
            surface,
            config,
            compositor_gpu,
            atlas,
        )
    }

    #[cfg(target_family = "wasm")]
    pub fn new_from_canvas(
        context: &WgpuContext,
        canvas: &web_sys::HtmlCanvasElement,
        config: WgpuSurfaceConfig,
    ) -> anyhow::Result<Self> {
        let surface = context
            .instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| anyhow::anyhow!("Failed to create surface: {e}"))?;
        Self::new_from_surface(context, surface, config)
    }

    #[cfg(target_family = "wasm")]
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new_from_surface(
        context: &WgpuContext,
        surface: wgpu::Surface<'static>,
        config: WgpuSurfaceConfig,
    ) -> anyhow::Result<Self> {
        let atlas = Arc::new(WgpuAtlas::from_context(context));
        Self::new_internal(None, context, surface, config, None, atlas)
    }

    fn new_internal(
        gpu_context: Option<GpuContext>,
        context: &WgpuContext,
        surface: wgpu::Surface<'static>,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
        atlas: Arc<WgpuAtlas>,
    ) -> anyhow::Result<Self> {
        let surface_caps = surface.get_capabilities(&context.adapter);
        let preferred_formats = [
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ];
        let surface_format = preferred_formats
            .iter()
            .find(|f| surface_caps.formats.contains(f))
            .copied()
            .or_else(|| surface_caps.formats.iter().find(|f| !f.is_srgb()).copied())
            .or_else(|| surface_caps.formats.first().copied())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Surface reports no supported texture formats for adapter {:?}",
                    context.adapter.get_info().name
                )
            })?;

        let pick_alpha_mode =
            |preferences: &[wgpu::CompositeAlphaMode]| -> anyhow::Result<wgpu::CompositeAlphaMode> {
                preferences
                    .iter()
                    .find(|p| surface_caps.alpha_modes.contains(p))
                    .copied()
                    .or_else(|| surface_caps.alpha_modes.first().copied())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Surface reports no supported alpha modes for adapter {:?}",
                            context.adapter.get_info().name
                        )
                    })
            };

        let transparent_alpha_mode = pick_alpha_mode(&[
            wgpu::CompositeAlphaMode::PreMultiplied,
            wgpu::CompositeAlphaMode::Inherit,
        ])?;

        let opaque_alpha_mode = pick_alpha_mode(&[
            wgpu::CompositeAlphaMode::Opaque,
            wgpu::CompositeAlphaMode::Inherit,
        ])?;

        let alpha_mode = if config.transparent {
            transparent_alpha_mode
        } else {
            opaque_alpha_mode
        };

        let device = Arc::clone(&context.device);
        let max_texture_size = device.limits().max_texture_dimension_2d;

        let requested_width = config.size.width.0 as u32;
        let requested_height = config.size.height.0 as u32;
        let clamped_width = requested_width.min(max_texture_size);
        let clamped_height = requested_height.min(max_texture_size);

        if clamped_width != requested_width || clamped_height != requested_height {
            warn!(
                "Requested surface size ({}, {}) exceeds maximum texture dimension {}. \
                 Clamping to ({}, {}). Window content may not fill the entire window.",
                requested_width, requested_height, max_texture_size, clamped_width, clamped_height
            );
        }

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: clamped_width.max(1),
            height: clamped_height.max(1),
            present_mode: config
                .preferred_present_mode
                .filter(|mode| surface_caps.present_modes.contains(mode))
                .unwrap_or(wgpu::PresentMode::Fifo),
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![],
        };
        // Configure the surface immediately. The adapter selection process already validated
        // that this adapter can successfully configure this surface.
        surface.configure(&context.device, &surface_config);

        let queue = Arc::clone(&context.queue);
        let rendering_params = RenderingParameters::new(&context.adapter, surface_format);
        let uses_webgl_instance_data = context.uses_webgl_instance_data();
        let dual_source_blending =
            context.supports_dual_source_blending() && !uses_webgl_instance_data;
        let bind_group_layouts = Self::create_bind_group_layouts(&device, uses_webgl_instance_data);
        let pipelines = Self::create_pipelines(
            &device,
            &bind_group_layouts,
            surface_format,
            alpha_mode,
            rendering_params.path_sample_count,
            dual_source_blending,
            uses_webgl_instance_data,
        );

        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let globals_size = std::mem::size_of::<GlobalParams>() as u64;
        let gamma_size = std::mem::size_of::<GammaParams>() as u64;
        let path_globals_offset = globals_size.next_multiple_of(uniform_alignment);
        let gamma_offset = (path_globals_offset + globals_size).next_multiple_of(uniform_alignment);

        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals_buffer"),
            size: gamma_offset + gamma_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (
            instance_data,
            instance_data_capacity,
            max_instance_data_size,
            instance_data_alignment,
        ) = if uses_webgl_instance_data {
            let max_texture_dimension = device.limits().max_texture_dimension_2d;
            let max_instance_data_size = (u64::from(max_texture_dimension).pow(2)
                * INSTANCE_TEXTURE_TEXEL_SIZE)
                .min(MAX_INSTANCE_BUFFER_SIZE);
            let initial_capacity = (2 * 1024 * 1024).min(max_instance_data_size);
            let (instance_data, capacity) =
                Self::create_instance_texture(&device, initial_capacity, max_texture_dimension);
            (
                instance_data,
                capacity,
                max_instance_data_size,
                INSTANCE_TEXTURE_TEXEL_SIZE,
            )
        } else {
            // Every frame allocation is exposed as one storage-buffer binding, so
            // its backing buffer must satisfy both the allocation and binding limits.
            let max_buffer_size = device
                .limits()
                .max_buffer_size
                .min(device.limits().max_storage_buffer_binding_size)
                .min(MAX_INSTANCE_BUFFER_SIZE);
            let initial_capacity = (2 * 1024 * 1024).min(max_buffer_size);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instance_buffer"),
                size: initial_capacity,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (
                InstanceData::Storage(buffer),
                initial_capacity,
                max_buffer_size,
                device.limits().min_storage_buffer_offset_alignment as u64,
            )
        };

        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals_bind_group"),
            layout: &bind_group_layouts.globals,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(globals_size).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: gamma_offset,
                        size: Some(NonZeroU64::new(gamma_size).unwrap()),
                    }),
                },
            ],
        });

        let path_globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("path_globals_bind_group"),
            layout: &bind_group_layouts.globals,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: path_globals_offset,
                        size: Some(NonZeroU64::new(globals_size).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: gamma_offset,
                        size: Some(NonZeroU64::new(gamma_size).unwrap()),
                    }),
                },
            ],
        });

        let adapter_info = context.adapter.get_info();

        let last_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let last_error_clone = Arc::clone(&last_error);
        device.on_uncaptured_error(Arc::new(move |error| {
            let mut guard = last_error_clone.lock().unwrap();
            *guard = Some(error.to_string());
        }));

        let resources = WgpuResources {
            device,
            queue,
            surface,
            pipelines,
            bind_group_layouts,
            atlas_sampler,
            globals_buffer,
            globals_bind_group,
            path_globals_bind_group,
            instance_data,
            // Defer intermediate texture creation to first draw call via ensure_intermediate_textures().
            // This avoids panics when the device/surface is in an invalid state during initialization.
            path_intermediate_texture: None,
            path_intermediate_view: None,
            path_msaa_texture: None,
            path_msaa_view: None,
        };

        Ok(Self {
            context: gpu_context,
            compositor_gpu,
            resources: Some(resources),
            surface_config,
            atlas,
            path_globals_offset,
            gamma_offset,
            instance_data_capacity,
            max_instance_data_size,
            instance_data_alignment,
            uses_webgl_instance_data,
            rendering_params,
            is_bgr: false,
            dual_source_blending,
            adapter_info,
            transparent_alpha_mode,
            opaque_alpha_mode,
            max_texture_size,
            last_error,
            failed_frame_count: 0,
            device_lost: context.device_lost_flag(),
            surface_configured: true,
            needs_redraw: false,
            external_frames: ExternalFrameState::default(),
            last_external_frame_outcome: None,
        })
    }

    fn create_bind_group_layouts(
        device: &wgpu::Device,
        uses_webgl_instance_data: bool,
    ) -> WgpuBindGroupLayouts {
        let globals =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("globals_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<GlobalParams>() as u64
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<GammaParams>() as u64
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let instance_data_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: if uses_webgl_instance_data {
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }
            } else {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            },
            count: None,
        };

        let instances = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instances_layout"),
            entries: &[instance_data_entry],
        });

        let texture = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let surfaces = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surfaces_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<SurfaceParams>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        WgpuBindGroupLayouts {
            globals,
            instances,
            texture,
            surfaces,
        }
    }

    fn create_instance_texture(
        device: &wgpu::Device,
        requested_capacity: u64,
        max_texture_dimension: u32,
    ) -> (InstanceData, u64) {
        let texel_count = requested_capacity.div_ceil(INSTANCE_TEXTURE_TEXEL_SIZE);
        let width = texel_count.min(u64::from(max_texture_dimension)).max(1) as u32;
        let height = texel_count
            .div_ceil(u64::from(width))
            .min(u64::from(max_texture_dimension))
            .max(1) as u32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("instance_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let capacity = u64::from(width) * u64::from(height) * INSTANCE_TEXTURE_TEXEL_SIZE;
        (
            InstanceData::Texture {
                texture,
                view,
                width,
                height,
            },
            capacity,
        )
    }

    fn create_pipelines(
        device: &wgpu::Device,
        layouts: &WgpuBindGroupLayouts,
        surface_format: wgpu::TextureFormat,
        alpha_mode: wgpu::CompositeAlphaMode,
        path_sample_count: u32,
        dual_source_blending: bool,
        uses_webgl_instance_data: bool,
    ) -> WgpuPipelines {
        // Diagnostic guard: verify the device actually has
        // DUAL_SOURCE_BLENDING. We have a crash report (ZED-5G1) where a
        // feature mismatch caused a wgpu-hal abort, but we haven't
        // identified the code path that produces the mismatch. This
        // guard prevents the crash and logs more evidence.
        // Remove this check once:
        // a) We find and fix the root cause, or
        // b) There are no reports of this warning appearing for some time.
        let device_has_feature = device
            .features()
            .contains(wgpu::Features::DUAL_SOURCE_BLENDING);
        if dual_source_blending && !device_has_feature {
            log::error!(
                "BUG: dual_source_blending flag is true but device does not \
                 have DUAL_SOURCE_BLENDING enabled (device features: {:?}). \
                 Falling back to mono text rendering. Please report this at \
                 https://github.com/zed-industries/zed/issues",
                device.features(),
            );
        }
        let dual_source_blending =
            dual_source_blending && device_has_feature && !uses_webgl_instance_data;

        let shader_source = if uses_webgl_instance_data {
            WEBGL_SHADERS
        } else {
            STORAGE_BUFFER_SHADERS
        };
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpui_shaders"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let subpixel_shader_module = if dual_source_blending {
            Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpui_subpixel_shaders"),
                source: wgpu::ShaderSource::Wgsl(SUBPIXEL_SHADERS.into()),
            }))
        } else {
            None
        };

        let blend_mode = match alpha_mode {
            wgpu::CompositeAlphaMode::PreMultiplied => {
                wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
            }
            _ => wgpu::BlendState::ALPHA_BLENDING,
        };

        let color_target = wgpu::ColorTargetState {
            format: surface_format,
            blend: Some(blend_mode),
            write_mask: wgpu::ColorWrites::ALL,
        };

        let create_pipeline = |name: &str,
                               vs_entry: &str,
                               fs_entry: &str,
                               globals_layout: &wgpu::BindGroupLayout,
                               data_layout: &wgpu::BindGroupLayout,
                               texture_layout: Option<&wgpu::BindGroupLayout>,
                               topology: wgpu::PrimitiveTopology,
                               color_targets: &[Option<wgpu::ColorTargetState>],
                               sample_count: u32,
                               module: &wgpu::ShaderModule| {
            let mut bind_group_layouts = vec![Some(globals_layout), Some(data_layout)];
            bind_group_layouts.extend(texture_layout.map(Some));
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{name}_layout")),
                bind_group_layouts: &bind_group_layouts,
                immediate_size: 0,
            });

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(vs_entry),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(fs_entry),
                    targets: color_targets,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };

        let quads = create_pipeline(
            "quads",
            "vs_quad",
            "fs_quad",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let shadows = create_pipeline(
            "shadows",
            "vs_shadow",
            "fs_shadow",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let path_rasterization = create_pipeline(
            "path_rasterization",
            "vs_path_rasterization",
            "fs_path_rasterization",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleList,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            path_sample_count,
            &shader_module,
        );

        let paths_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let paths = create_pipeline(
            "paths",
            "vs_path",
            "fs_path",
            &layouts.globals,
            &layouts.instances,
            Some(&layouts.texture),
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(paths_blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            1,
            &shader_module,
        );

        let underlines = create_pipeline(
            "underlines",
            "vs_underline",
            "fs_underline",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let mono_sprites = create_pipeline(
            "mono_sprites",
            "vs_mono_sprite",
            "fs_mono_sprite",
            &layouts.globals,
            &layouts.instances,
            Some(&layouts.texture),
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let subpixel_sprites = if let Some(subpixel_module) = &subpixel_shader_module {
            let subpixel_blend = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Src1,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc1,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            };

            Some(create_pipeline(
                "subpixel_sprites",
                "vs_subpixel_sprite",
                "fs_subpixel_sprite",
                &layouts.globals,
                &layouts.instances,
                Some(&layouts.texture),
                wgpu::PrimitiveTopology::TriangleStrip,
                &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(subpixel_blend),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
                1,
                subpixel_module,
            ))
        } else {
            None
        };

        let poly_sprites = create_pipeline(
            "poly_sprites",
            "vs_poly_sprite",
            "fs_poly_sprite",
            &layouts.globals,
            &layouts.instances,
            Some(&layouts.texture),
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let surfaces = create_pipeline(
            "surfaces",
            "vs_surface",
            "fs_surface",
            &layouts.globals,
            &layouts.surfaces,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let surface_rgba = create_pipeline(
            "surface_rgba",
            "vs_surface_rgba",
            "fs_surface_rgba",
            &layouts.globals,
            &layouts.instances_with_texture,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target)],
            1,
            &shader_module,
        );

        WgpuPipelines {
            quads,
            shadows,
            path_rasterization,
            paths,
            underlines,
            mono_sprites,
            subpixel_sprites,
            poly_sprites,
            surfaces,
            surface_rgba,
        }
    }

    fn create_path_intermediate(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_intermediate"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_msaa_if_needed(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Option<(wgpu::Texture, wgpu::TextureView)> {
        if sample_count <= 1 {
            return None;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_msaa"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Some((texture, view))
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        let width = size.width.0 as u32;
        let height = size.height.0 as u32;

        if width != self.surface_config.width || height != self.surface_config.height {
            let clamped_width = width.min(self.max_texture_size);
            let clamped_height = height.min(self.max_texture_size);

            if clamped_width != width || clamped_height != height {
                warn!(
                    "Requested surface size ({}, {}) exceeds maximum texture dimension {}. \
                     Clamping to ({}, {}). Window content may not fill the entire window.",
                    width, height, self.max_texture_size, clamped_width, clamped_height
                );
            }

            self.surface_config.width = clamped_width.max(1);
            self.surface_config.height = clamped_height.max(1);
            let surface_config = self.surface_config.clone();

            let Some(resources) = self.resources.as_mut() else {
                return;
            };

            // Wait for any in-flight GPU work to complete before destroying textures
            if let Err(e) = resources.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            }) {
                warn!("Failed to poll device during resize: {e:?}");
            }

            // Destroy old textures before allocating new ones to avoid GPU memory spikes
            if let Some(ref texture) = resources.path_intermediate_texture {
                texture.destroy();
            }
            if let Some(ref texture) = resources.path_msaa_texture {
                texture.destroy();
            }

            resources
                .surface
                .configure(&resources.device, &surface_config);

            // Invalidate intermediate textures - they will be lazily recreated
            // in draw() after we confirm the surface is healthy. This avoids
            // panics when the device/surface is in an invalid state during resize.
            resources.invalidate_intermediate_textures();
        }
    }

    fn ensure_intermediate_textures(&mut self) {
        if self.resources().path_intermediate_texture.is_some() {
            return;
        }

        let format = self.surface_config.format;
        let width = self.surface_config.width;
        let height = self.surface_config.height;
        let path_sample_count = self.rendering_params.path_sample_count;
        let resources = self.resources_mut();

        let (t, v) = Self::create_path_intermediate(&resources.device, format, width, height);
        resources.path_intermediate_texture = Some(t);
        resources.path_intermediate_view = Some(v);

        let (path_msaa_texture, path_msaa_view) = Self::create_msaa_if_needed(
            &resources.device,
            format,
            width,
            height,
            path_sample_count,
        )
        .map(|(t, v)| (Some(t), Some(v)))
        .unwrap_or((None, None));
        resources.path_msaa_texture = path_msaa_texture;
        resources.path_msaa_view = path_msaa_view;
    }

    pub fn set_subpixel_layout(&mut self, is_bgr: bool) {
        self.is_bgr = is_bgr;
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        let new_alpha_mode = if transparent {
            self.transparent_alpha_mode
        } else {
            self.opaque_alpha_mode
        };

        if new_alpha_mode != self.surface_config.alpha_mode {
            self.surface_config.alpha_mode = new_alpha_mode;
            let surface_config = self.surface_config.clone();
            let path_sample_count = self.rendering_params.path_sample_count;
            let dual_source_blending = self.dual_source_blending;
            let uses_webgl_instance_data = self.uses_webgl_instance_data;
            let Some(resources) = self.resources.as_mut() else {
                return;
            };
            resources
                .surface
                .configure(&resources.device, &surface_config);
            resources.pipelines = Self::create_pipelines(
                &resources.device,
                &resources.bind_group_layouts,
                surface_config.format,
                surface_config.alpha_mode,
                path_sample_count,
                dual_source_blending,
                uses_webgl_instance_data,
            );
        }
    }

    #[allow(dead_code)]
    pub fn viewport_size(&self) -> Size<DevicePixels> {
        Size {
            width: DevicePixels(self.surface_config.width as i32),
            height: DevicePixels(self.surface_config.height as i32),
        }
    }

    pub fn sprite_atlas(&self) -> &Arc<WgpuAtlas> {
        &self.atlas
    }

    pub fn supports_dual_source_blending(&self) -> bool {
        self.dual_source_blending
    }

    pub fn gpu_specs(&self) -> GpuSpecs {
        GpuSpecs {
            is_software_emulated: self.adapter_info.device_type == wgpu::DeviceType::Cpu,
            device_name: self.adapter_info.name.clone(),
            driver_name: self.adapter_info.driver.clone(),
            driver_info: self.adapter_info.driver_info.clone(),
        }
    }

    pub fn max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    pub fn draw(&mut self, scene: &Scene) -> bool {
        #[cfg(target_family = "wasm")]
        if self.device_lost() {
            if self.surface_configured {
                log::error!(
                    "Browser graphics context was lost; rendering has stopped. Reload the page to recover."
                );
                self.surface_configured = false;
            }
            return false;
        }

        // Bail out early if the surface has been unconfigured (e.g. during
        // Android background/rotation transitions).  Attempting to acquire
        // a texture from an unconfigured surface can block indefinitely on
        // some drivers (Adreno).
        if !self.surface_configured {
            return false;
        }

        let last_error = self.last_error.lock().unwrap().take();
        if let Some(error) = last_error {
            self.failed_frame_count += 1;
            log::error!(
                "GPU error during frame (failure {} of 10): {error}",
                self.failed_frame_count
            );

            // TBD. Does retrying more actually help?
            if self.failed_frame_count > 10 {
                panic!("Too many consecutive GPU errors. Last error: {error}");
            } else if self.failed_frame_count > 5 {
                if let Some(res) = self.resources.as_mut() {
                    res.invalidate_intermediate_textures();
                }
                self.atlas.clear();
                self.needs_redraw = true;
                self.failed_frame_count = 0;
                return false;
            }
        } else {
            self.failed_frame_count = 0;
        }

        self.atlas.before_frame();

        let frame = match self.resources().surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                // Textures must be destroyed before the surface can be reconfigured.
                drop(frame);
                let surface_config = self.surface_config.clone();
                let resources = self.resources_mut();
                resources
                    .surface
                    .configure(&resources.device, &surface_config);
                return false;
            }
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                let surface_config = self.surface_config.clone();
                let resources = self.resources_mut();
                resources
                    .surface
                    .configure(&resources.device, &surface_config);
                return false;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return false;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                *self.last_error.lock().unwrap() =
                    Some("Surface texture validation error".to_string());
                return false;
            }
        };

        // Now that we know the surface is healthy, ensure intermediate textures exist
        self.ensure_intermediate_textures();

        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let gamma_params = GammaParams {
            gamma_ratios: self.rendering_params.gamma_ratios,
            grayscale_enhanced_contrast: self.rendering_params.grayscale_enhanced_contrast,
            subpixel_enhanced_contrast: self.rendering_params.subpixel_enhanced_contrast,
            is_bgr: self.is_bgr as u32,
            _pad: 0,
        };

        let globals = GlobalParams {
            viewport_size: [
                self.surface_config.width as f32,
                self.surface_config.height as f32,
            ],
            premultiplied_alpha: if self.surface_config.alpha_mode
                == wgpu::CompositeAlphaMode::PreMultiplied
            {
                1
            } else {
                0
            },
            pad: 0,
        };

        let path_globals = GlobalParams {
            premultiplied_alpha: 0,
            ..globals
        };

        {
            let resources = self.resources();
            resources.queue.write_buffer(
                &resources.globals_buffer,
                0,
                bytemuck::bytes_of(&globals),
            );
            resources.queue.write_buffer(
                &resources.globals_buffer,
                self.path_globals_offset,
                bytemuck::bytes_of(&path_globals),
            );
            resources.queue.write_buffer(
                &resources.globals_buffer,
                self.gamma_offset,
                bytemuck::bytes_of(&gamma_params),
            );
        }

        if let Err(error) = self.record_frame(scene, &frame_view) {
            log::error!("{error:#}");
            self.resources().queue.submit(std::iter::empty());
            return false;
        }

        self.resources().queue.present(frame);
        true
    }

    fn record_frame(&mut self, scene: &Scene, frame_view: &wgpu::TextureView) -> Result<()> {
        let queue = Arc::clone(&self.resources().queue);
        let mut instance_offset = 0;
        let instance_bindings = self
            .write_instances(scene, &mut instance_offset)
            .with_context(|| {
                format!(
                    "scene too large: {} paths, {} shadows, {} quads, {} underlines, {} monochrome sprites, {} subpixel sprites, {} polychrome sprites",
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.subpixel_sprites.len(),
                    scene.polychrome_sprites.len(),
                )
            })?;

        let mut encoder =
            self.resources()
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("main_encoder"),
                });

        #[cfg(target_os = "linux")]
        let external_selection = {
            let context = self.gpu_context_handle();
            let outcome = self.external_frames.prepare_frame(context.as_ref());
            if outcome != ExternalFrameOutcome::Accepted {
                self.last_external_frame_outcome = Some(outcome);
                anyhow::bail!("external frame preparation failed");
            }
            self.external_frames.selection_for_scene(scene)
        };
        #[cfg(target_os = "linux")]
        let (mut acquire_encoder, mut release_encoder) = {
            if let Some(selection) = external_selection.filter(|slot| self.external_frames.selected_is_external(*slot)) {
                let mut acquire = self.resources().device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("external_acquire_encoder") });
                let release = self.resources().device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("external_release_encoder") });
                let outcome = self.external_frames.encode_acquire(selection, &mut acquire);
                if outcome != ExternalFrameOutcome::Accepted {
                    self.last_external_frame_outcome = Some(outcome);
                    anyhow::bail!("external frame acquire encoding failed");
                }
                (Some(acquire), Some(release))
            } else {
                (None, None)
            }
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            for batch in scene.batches() {
                match batch {
                    PrimitiveBatch::Quads(range) => self.draw_instances(
                        &instance_bindings.quads,
                        &self.resources().pipelines.quads,
                        instance_range(range),
                        &mut pass,
                    ),
                    PrimitiveBatch::Shadows(range) => self.draw_instances(
                        &instance_bindings.shadows,
                        &self.resources().pipelines.shadows,
                        instance_range(range),
                        &mut pass,
                    ),
                    PrimitiveBatch::Paths(range) => {
                        let paths = &scene.paths[range];
                        if paths.is_empty() {
                            continue;
                        }

                        drop(pass);
                        let rasterized = self.draw_paths_to_intermediate(
                            &mut encoder,
                            paths,
                            &mut instance_offset,
                        )?;

                        pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("main_pass_continued"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: frame_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            ..Default::default()
                        });

                        if rasterized {
                            self.draw_paths_from_intermediate(
                                paths,
                                &mut instance_offset,
                                &mut pass,
                            )?;
                        }
                    }
                    PrimitiveBatch::Underlines(range) => self.draw_instances(
                        &instance_bindings.underlines,
                        &self.resources().pipelines.underlines,
                        instance_range(range),
                        &mut pass,
                    ),
                    PrimitiveBatch::MonochromeSprites { texture_id, range } => self.draw_sprites(
                        &instance_bindings.monochrome_sprites,
                        texture_id,
                        &self.resources().pipelines.mono_sprites,
                        instance_range(range),
                        &mut pass,
                    ),
                    PrimitiveBatch::SubpixelSprites { texture_id, range } => {
                        let resources = self.resources();
                        self.draw_sprites(
                            &instance_bindings.subpixel_sprites,
                            texture_id,
                            resources
                                .pipelines
                                .subpixel_sprites
                                .as_ref()
                                .unwrap_or(&resources.pipelines.mono_sprites),
                            instance_range(range),
                            &mut pass,
                        );
                    }
                    PrimitiveBatch::PolychromeSprites { texture_id, range } => self.draw_sprites(
                        &instance_bindings.polychrome_sprites,
                        texture_id,
                        &self.resources().pipelines.poly_sprites,
                        instance_range(range),
                        &mut pass,
                    ),
                    PrimitiveBatch::Surfaces(range) => {
                        if !self.draw_surfaces(&scene.surfaces[range], &mut instance_offset, &mut pass) {
                            anyhow::bail!("surface instance buffer exhausted");
                        }
                    }
                }
            }
        }

        let render_command_buffer = encoder.finish();
        #[cfg(target_os = "linux")]
        if let (Some(selection), Some(acquire), Some(mut release)) = (external_selection, acquire_encoder.take(), release_encoder.take()) {
            let outcome = self.external_frames.encode_release(selection, &mut release);
            if outcome != ExternalFrameOutcome::Accepted {
                self.last_external_frame_outcome = Some(outcome);
                anyhow::bail!("external frame release encoding failed");
            }
            let acquire_commands = acquire.finish();
            let release_commands = release.finish();
            let outcome = self.external_frames.import_sync(selection, &queue);
            if outcome != ExternalFrameOutcome::Accepted {
                self.last_external_frame_outcome = Some(outcome);
                anyhow::bail!("external frame acquire preparation failed");
            }
            queue.submit([acquire_commands, render_command_buffer, release_commands]);
        } else {
            queue.submit(std::iter::once(render_command_buffer));
        }
        #[cfg(not(target_os = "linux"))]
        queue.submit(std::iter::once(render_command_buffer));
        #[cfg(target_os = "linux")]
        self.external_frames
            .commit_after_submission_with_queue(external_selection, &queue);
        #[cfg(not(target_os = "linux"))]
        self.external_frames.commit_after_submission();
        Ok(())
    }

    fn write_instances(
        &mut self,
        scene: &Scene,
        instance_offset: &mut u64,
    ) -> Result<InstanceBindings> {
        Ok(InstanceBindings {
            quads: self.write_instance_binding(
                "quads_bind_group",
                instance_offset,
                &scene.quads,
            )?,
            shadows: self.write_instance_binding(
                "shadows_bind_group",
                instance_offset,
                &scene.shadows,
            )?,
            underlines: self.write_instance_binding(
                "underlines_bind_group",
                instance_offset,
                &scene.underlines,
            )?,
            monochrome_sprites: self.write_instance_binding(
                "monochrome_sprites_bind_group",
                instance_offset,
                &scene.monochrome_sprites,
            )?,
            subpixel_sprites: self.write_instance_binding(
                "subpixel_sprites_bind_group",
                instance_offset,
                &scene.subpixel_sprites,
            )?,
            polychrome_sprites: self.write_instance_binding(
                "polychrome_sprites_bind_group",
                instance_offset,
                &scene.polychrome_sprites,
            )?,
        })
    }

    fn create_texture_bind_group(
        &self,
        label: &str,
        texture_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let resources = self.resources();
        resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &resources.bind_group_layouts.texture,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&resources.atlas_sampler),
                    },
                ],
            })
    }

    fn draw_surfaces(
        &self,
        surfaces: &[gpui::PaintSurface],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        struct RgbaDraw {
            bind_group: wgpu::BindGroup,
            scissor_rect: (u32, u32, u32, u32),
            _view: wgpu::TextureView,
        }

        let mut rgba_draws: Vec<RgbaDraw> = Vec::new();
        // NV12 data collected without per-surface buffers — the cached
        // uniform buffer is reused via interleaved write-then-draw.
        struct Nv12Data {
            y_view: wgpu::TextureView,
            cb_cr_view: wgpu::TextureView,
            params_data: Vec<u8>,
            scissor_rect: (u32, u32, u32, u32),
        }
        let mut nv12_items: Vec<Nv12Data> = Vec::new();

        for surface in surfaces {
            // Skip zero-sized surfaces (collapsed panels, hidden elements).
            if surface.bounds.size.width.0 <= 0.0 || surface.bounds.size.height.0 <= 0.0 {
                continue;
            }

            let resources = self.resources();

            match &surface.content {
                gpui::SurfaceContent::WgpuTexture(texture) => {
                    let instance = SurfaceInstance {
                        bounds: surface.bounds.into(),
                        content_mask: surface.content_mask.bounds.into(),
                    };
                    let data = unsafe {
                        std::slice::from_raw_parts(
                            &instance as *const SurfaceInstance as *const u8,
                            std::mem::size_of::<SurfaceInstance>(),
                        )
                    };
                    let Some((offset, size)) = self.write_to_instance_buffer(instance_offset, data)
                    else {
                        return false;
                    };

                    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                    let bind_group =
                        resources
                            .device
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("surface_rgba_bind_group"),
                                layout: &resources.bind_group_layouts.instances_with_texture,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: self.instance_binding(offset, size),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::TextureView(&view),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: wgpu::BindingResource::Sampler(
                                            &resources.atlas_sampler,
                                        ),
                                    },
                                ],
                            });

                    let scissor_rect = (
                        surface.content_mask.bounds.origin.x.0.max(0.0) as u32,
                        surface.content_mask.bounds.origin.y.0.max(0.0) as u32,
                        surface.content_mask.bounds.size.width.0.max(0.0) as u32,
                        surface.content_mask.bounds.size.height.0.max(0.0) as u32,
                    );

                    rgba_draws.push(RgbaDraw {
                        bind_group,
                        scissor_rect,
                        _view: view,
                    });
                    // write_to_instance_buffer already advanced instance_offset
                    // to the next aligned position; do not add extra offset here.
                }
                gpui::SurfaceContent::WgpuTextureNv12 { .. }
                | gpui::SurfaceContent::WgpuTextureNv12WithColorTransform { .. } => {
                    let (y_texture, cb_cr_texture, color_transform) = match &surface.content {
                        gpui::SurfaceContent::WgpuTextureNv12 {
                            y_texture,
                            cb_cr_texture,
                            ..
                        } => (
                            y_texture,
                            cb_cr_texture,
                            gpui::Nv12ColorTransform::default(),
                        ),
                        gpui::SurfaceContent::WgpuTextureNv12WithColorTransform {
                            y_texture,
                            cb_cr_texture,
                            color_transform,
                            ..
                        } => (y_texture, cb_cr_texture, *color_transform),
                        #[allow(unreachable_patterns)]
                        _ => continue,
                    };
                    let params = SurfaceParams::new(
                        surface.bounds.into(),
                        surface.content_mask.bounds.into(),
                        color_transform,
                    );
                    let params_data = unsafe {
                        std::slice::from_raw_parts(
                            &params as *const SurfaceParams as *const u8,
                            std::mem::size_of::<SurfaceParams>(),
                        )
                    };

                    let y_view = y_texture.create_view(&wgpu::TextureViewDescriptor::default());
                    let cb_cr_view =
                        cb_cr_texture.create_view(&wgpu::TextureViewDescriptor::default());

                    nv12_items.push(Nv12Data {
                        y_view,
                        cb_cr_view,
                        params_data: params_data.to_vec(),
                        scissor_rect: (
                            surface.content_mask.bounds.origin.x.0.max(0.0) as u32,
                            surface.content_mask.bounds.origin.y.0.max(0.0) as u32,
                            surface.content_mask.bounds.size.width.0.max(0.0) as u32,
                            surface.content_mask.bounds.size.height.0.max(0.0) as u32,
                        ),
                    });
                }
                gpui::SurfaceContent::WgpuTextureNv12Multiplanar {
                    texture,
                    color_transform,
                    ..
                } => {
                    let params = SurfaceParams::new(
                        surface.bounds.into(),
                        surface.content_mask.bounds.into(),
                        *color_transform,
                    );
                    let params_data = unsafe {
                        std::slice::from_raw_parts(
                            &params as *const SurfaceParams as *const u8,
                            std::mem::size_of::<SurfaceParams>(),
                        )
                    };
                    let Some(plane0_descriptor) =
                        nv12_plane_view_descriptor(wgpu::TextureAspect::Plane0)
                    else {
                        return false;
                    };
                    let Some(plane1_descriptor) =
                        nv12_plane_view_descriptor(wgpu::TextureAspect::Plane1)
                    else {
                        return false;
                    };
                    let y_view = texture.create_view(&plane0_descriptor);
                    let cb_cr_view = texture.create_view(&plane1_descriptor);
                    nv12_items.push(Nv12Data {
                        y_view,
                        cb_cr_view,
                        params_data: params_data.to_vec(),
                        scissor_rect: (
                            surface.content_mask.bounds.origin.x.0.max(0.0) as u32,
                            surface.content_mask.bounds.origin.y.0.max(0.0) as u32,
                            surface.content_mask.bounds.size.width.0.max(0.0) as u32,
                            surface.content_mask.bounds.size.height.0.max(0.0) as u32,
                        ),
                    });
                }
                #[allow(unreachable_patterns)]
                _ => continue,
            }
        }

        let resources = self.resources();

        // Draw RGBA surfaces with the passthrough pipeline.
        for draw in &rgba_draws {
            pass.set_pipeline(&resources.pipelines.surface_rgba);
            pass.set_bind_group(0, &resources.globals_bind_group, &[]);
            pass.set_bind_group(1, &draw.bind_group, &[]);
            pass.set_scissor_rect(
                draw.scissor_rect.0,
                draw.scissor_rect.1,
                draw.scissor_rect.2,
                draw.scissor_rect.3,
            );
            pass.draw(0..4, 0..1);
        }

        // Draw NV12 surfaces with the YUV→RGB conversion pipeline.
        // Each surface uses its own uniform buffer (created here,
        // not cached) so bind-group lifetimes are correct and
        // buffer offsets are naturally aligned.
        struct Nv12Draw {
            bind_group: wgpu::BindGroup,
            _uniform_buffer: wgpu::Buffer,
            scissor_rect: (u32, u32, u32, u32),
        }
        let mut nv12_draws: Vec<Nv12Draw> = Vec::with_capacity(nv12_items.len());

        for item in nv12_items {
            let uniform_buffer = resources.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("surface_nv12_uniform"),
                size: std::mem::size_of::<SurfaceParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            resources
                .queue
                .write_buffer(&uniform_buffer, 0, &item.params_data);

            let bind_group = resources
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("surface_nv12_bind_group"),
                    layout: &resources.bind_group_layouts.surfaces,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&item.y_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&item.cb_cr_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&resources.atlas_sampler),
                        },
                    ],
                });

            nv12_draws.push(Nv12Draw {
                bind_group,
                _uniform_buffer: uniform_buffer,
                scissor_rect: item.scissor_rect,
            });
        }

        for draw in &nv12_draws {
            pass.set_pipeline(&resources.pipelines.surfaces);
            pass.set_bind_group(0, &resources.globals_bind_group, &[]);
            pass.set_bind_group(1, &draw.bind_group, &[]);
            pass.set_scissor_rect(
                draw.scissor_rect.0,
                draw.scissor_rect.1,
                draw.scissor_rect.2,
                draw.scissor_rect.3,
            );
            pass.draw(0..4, 0..1);
        }

        true
    }

    fn draw_instances(
        &self,
        instances: &InstanceBinding,
        pipeline: &wgpu::RenderPipeline,
        range: Range<u32>,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        if range.is_empty() {
            return;
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.resources().globals_bind_group, &[]);
        pass.set_bind_group(1, &instances.bind_group, &[]);
        pass.draw(
            0..4,
            instances.first_instance + range.start..instances.first_instance + range.end,
        );
    }

    fn draw_sprites(
        &self,
        sprite_instances: &InstanceBinding,
        texture_id: AtlasTextureId,
        pipeline: &wgpu::RenderPipeline,
        range: Range<u32>,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        if range.is_empty() {
            return;
        }
        let texture_info = self.atlas.get_texture_info(texture_id);
        let texture =
            self.create_texture_bind_group("atlas_texture_bind_group", &texture_info.view);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.resources().globals_bind_group, &[]);
        pass.set_bind_group(1, &sprite_instances.bind_group, &[]);
        pass.set_bind_group(2, &texture, &[]);
        pass.draw(
            0..4,
            sprite_instances.first_instance + range.start
                ..sprite_instances.first_instance + range.end,
        );
    }

    unsafe fn instance_bytes<T>(instances: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                instances.as_ptr() as *const u8,
                std::mem::size_of_val(instances),
            )
        }
    }

    fn draw_paths_from_intermediate(
        &mut self,
        paths: &[Path<ScaledPixels>],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> Result<()> {
        let first_path = &paths[0];
        let sprites: Vec<PathSprite> = if paths.last().map(|p| &p.order) == Some(&first_path.order)
        {
            paths
                .iter()
                .map(|p| PathSprite {
                    bounds: p.clipped_bounds(),
                })
                .collect()
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            vec![PathSprite { bounds }]
        };

        let Some(path_intermediate_view) = self.resources().path_intermediate_view.clone() else {
            return Ok(());
        };
        let instances =
            self.write_instance_binding("path_sprites_bind_group", instance_offset, &sprites)?;
        let texture = self.create_texture_bind_group(
            "path_intermediate_texture_bind_group",
            &path_intermediate_view,
        );
        let resources = self.resources();
        pass.set_pipeline(&resources.pipelines.paths);
        pass.set_bind_group(0, &resources.globals_bind_group, &[]);
        pass.set_bind_group(1, &instances.bind_group, &[]);
        pass.set_bind_group(2, &texture, &[]);
        pass.draw(
            0..4,
            instances.first_instance..instances.first_instance + sprites.len() as u32,
        );
        Ok(())
    }

    fn draw_paths_to_intermediate(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        paths: &[Path<ScaledPixels>],
        instance_offset: &mut u64,
    ) -> Result<bool> {
        let mut vertices = Vec::new();
        for path in paths {
            let bounds = path.clipped_bounds();
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds,
            }));
        }

        if vertices.is_empty() {
            return Ok(false);
        }

        let vertex_binding = self.write_instance_binding(
            "path_rasterization_bind_group",
            instance_offset,
            &vertices,
        )?;

        let resources = self.resources();
        let Some(path_intermediate_view) = resources.path_intermediate_view.as_ref() else {
            return Ok(false);
        };

        let (target_view, resolve_target) = if let Some(ref msaa_view) = resources.path_msaa_view {
            (msaa_view, Some(path_intermediate_view))
        } else {
            (path_intermediate_view, None)
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("path_rasterization_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&resources.pipelines.path_rasterization);
            pass.set_bind_group(0, &resources.path_globals_bind_group, &[]);
            pass.set_bind_group(1, &vertex_binding.bind_group, &[]);
            // The path rasterization shader loads records by vertex index
            // rather than instance index, so the allocation's base shifts the
            // vertex range here.
            pass.draw(
                vertex_binding.first_instance
                    ..vertex_binding.first_instance + vertices.len() as u32,
                0..1,
            );
        }

        Ok(true)
    }

    fn write_instance_binding<T>(
        &mut self,
        label: &str,
        instance_offset: &mut u64,
        instances: &[T],
    ) -> Result<InstanceBinding> {
        let data = unsafe { Self::instance_bytes(instances) };
        // wgpu rejects zero-sized bindings, so empty primitive arrays still
        // reserve the 16-byte minimum.
        let size = (data.len() as u64).max(16);
        let stride = (std::mem::size_of::<T>() as u64).max(1);
        let (alignment, allocation_size) = if self.uses_webgl_instance_data {
            // The texture transport has no binding offset: the shader indexes
            // the instance texture absolutely, so each allocation must start on
            // a whole instance (a stride multiple) and a whole texel, and must
            // end on a texel boundary so the zero padding of its final partial
            // texel cannot overlap the next allocation.
            (
                least_common_multiple(self.instance_data_alignment, stride),
                size.next_multiple_of(INSTANCE_TEXTURE_TEXEL_SIZE),
            )
        } else {
            (self.instance_data_alignment.max(1), size)
        };
        let mut offset = (*instance_offset).next_multiple_of(alignment);
        if offset + allocation_size > self.instance_data_capacity {
            self.grow_instance_data(allocation_size)?;
            offset = 0;
        }
        *instance_offset = offset + allocation_size;

        let first_instance = if self.uses_webgl_instance_data {
            u32::try_from(offset / stride).context("instance index exceeds u32 range")?
        } else {
            0
        };

        let resources = self.resources();
        if !data.is_empty() {
            match &resources.instance_data {
                InstanceData::Storage(buffer) => resources.queue.write_buffer(buffer, offset, data),
                InstanceData::Texture { .. } => {
                    Self::write_instance_texture(resources, offset, data)
                }
            }
        }
        let bind_group = resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &resources.bind_group_layouts.instances,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: match &resources.instance_data {
                        InstanceData::Storage(buffer) => {
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer,
                                offset,
                                size: NonZeroU64::new(size),
                            })
                        }
                        InstanceData::Texture { view, .. } => {
                            wgpu::BindingResource::TextureView(view)
                        }
                    },
                }],
            });
        Ok(InstanceBinding {
            bind_group,
            first_instance,
        })
    }

    fn write_instance_texture(resources: &WgpuResources, offset: u64, data: &[u8]) {
        let InstanceData::Texture {
            texture,
            width,
            height,
            ..
        } = &resources.instance_data
        else {
            return;
        };
        let mut byte_offset = 0usize;
        let mut texel_offset = offset / INSTANCE_TEXTURE_TEXEL_SIZE;
        while byte_offset < data.len() {
            let x = (texel_offset % u64::from(*width)) as u32;
            let y = (texel_offset / u64::from(*width)) as u32;
            if y >= *height {
                // The capacity check in write_instance_binding should make this
                // unreachable. Truncating silently would leave stale bytes in the
                // texture and draw garbage for the remaining instances.
                debug_assert!(
                    false,
                    "instance texture write out of bounds: row {y} >= height {}",
                    *height
                );
                log::error!(
                    "instance texture write out of bounds; dropping {} bytes of instance data",
                    data.len() - byte_offset
                );
                return;
            }
            let available_texels = u64::from(*width - x);
            let remaining_bytes = data.len() - byte_offset;
            let complete_texels = remaining_bytes as u64 / INSTANCE_TEXTURE_TEXEL_SIZE;
            let texels = complete_texels.min(available_texels);
            if texels > 0 {
                let byte_count = (texels * INSTANCE_TEXTURE_TEXEL_SIZE) as usize;
                resources.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x, y, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &data[byte_offset..byte_offset + byte_count],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(byte_count as u32),
                        rows_per_image: None,
                    },
                    wgpu::Extent3d {
                        width: texels as u32,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
                byte_offset += byte_count;
                texel_offset += texels;
                continue;
            }

            let mut final_texel = [0; INSTANCE_TEXTURE_TEXEL_SIZE as usize];
            final_texel[..remaining_bytes].copy_from_slice(&data[byte_offset..]);
            resources.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                &final_texel,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(INSTANCE_TEXTURE_TEXEL_SIZE as u32),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            break;
        }
    }

    fn grow_instance_data(&mut self, required: u64) -> Result<()> {
        let capacity = (self.instance_data_capacity * 2)
            .max(required.next_power_of_two())
            .min(self.max_instance_data_size);
        anyhow::ensure!(
            capacity >= required,
            "instance data needs {required} bytes, above the maximum of {}",
            self.max_instance_data_size
        );
        anyhow::ensure!(
            capacity > self.instance_data_capacity,
            "frame instance data exceeds the {}-byte maximum",
            self.max_instance_data_size
        );
        log::debug!(
            "instance data grown from {} to {capacity}",
            self.instance_data_capacity
        );
        // Bind groups created earlier in the frame keep the previous buffer or
        // texture alive, so allocations written before the grow remain valid;
        // only subsequent writes land in the new allocation.
        let uses_webgl_instance_data = self.uses_webgl_instance_data;
        let resources = self.resources_mut();
        if uses_webgl_instance_data {
            let max_texture_dimension = resources.device.limits().max_texture_dimension_2d;
            let (instance_data, actual_capacity) =
                Self::create_instance_texture(&resources.device, capacity, max_texture_dimension);
            resources.instance_data = instance_data;
            self.instance_data_capacity = actual_capacity;
        } else {
            resources.instance_data =
                InstanceData::Storage(resources.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("instance_buffer"),
                    size: capacity,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            self.instance_data_capacity = capacity;
        }
        Ok(())
    }

    /// Mark the surface as unconfigured so rendering is skipped until a new
    /// surface is provided via [`replace_surface`](Self::replace_surface).
    ///
    /// This does **not** drop the renderer — the device, queue, atlas, and
    /// pipelines stay alive.  Use this when the native window is destroyed
    /// (e.g. Android `TerminateWindow`) but you intend to re-create the
    /// surface later without losing cached atlas textures.
    pub fn unconfigure_surface(&mut self) {
        self.surface_configured = false;
        // Drop intermediate textures since they reference the old surface size.
        if let Some(res) = self.resources.as_mut() {
            res.invalidate_intermediate_textures();
        }
    }

    /// Replace the wgpu surface with a new one (e.g. after Android destroys
    /// and recreates the native window).  Keeps the device, queue, atlas, and
    /// all pipelines intact so cached `AtlasTextureId`s remain valid.
    ///
    /// The `instance` **must** be the same [`wgpu::Instance`] that was used to
    /// create the adapter and device (i.e. from the [`WgpuContext`]).  Using a
    /// different instance will cause a "Device does not exist" panic because
    /// the wgpu device is bound to its originating instance.
    #[cfg(not(target_family = "wasm"))]
    pub fn replace_surface<W: HasWindowHandle>(
        &mut self,
        window: &W,
        config: WgpuSurfaceConfig,
        instance: &wgpu::Instance,
    ) -> anyhow::Result<()> {
        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;

        let surface = create_surface(instance, window_handle.as_raw())?;

        let width = (config.size.width.0 as u32).max(1);
        let height = (config.size.height.0 as u32).max(1);

        let alpha_mode = if config.transparent {
            self.transparent_alpha_mode
        } else {
            self.opaque_alpha_mode
        };

        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface_config.alpha_mode = alpha_mode;
        if let Some(mode) = config.preferred_present_mode {
            self.surface_config.present_mode = mode;
        }

        {
            let res = self
                .resources
                .as_mut()
                .expect("GPU resources not available");
            surface.configure(&res.device, &self.surface_config);
            res.surface = surface;

            // Invalidate intermediate textures — they'll be recreated lazily.
            res.invalidate_intermediate_textures();
        }

        self.surface_configured = true;

        Ok(())
    }

    pub fn destroy(&mut self) {
        // Release surface-bound GPU resources eagerly so the underlying native
        // window can be destroyed before the renderer itself is dropped.
        self.resources.take();
    }

    /// Returns true if the GPU device was lost and recovery is needed.
    pub fn device_lost(&self) -> bool {
        self.device_lost.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Returns true if a redraw is needed because GPU state was cleared.
    /// Calling this method clears the flag.
    pub fn needs_redraw(&mut self) -> bool {
        std::mem::take(&mut self.needs_redraw)
    }

    /// Recovers from a lost GPU device by recreating the renderer with a new context.
    ///
    /// Call this after detecting `device_lost()` returns true.
    ///
    /// This method coordinates recovery across multiple windows:
    /// - The first window to call this will recreate the shared context
    /// - Subsequent windows will adopt the already-recovered context
    #[cfg(not(target_family = "wasm"))]
    pub fn recover<W>(&mut self, window: &W) -> anyhow::Result<()>
    where
        W: HasWindowHandle + HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
    {
        let gpu_context = self.context.as_ref().expect("recover requires gpu_context");

        // Check if another window already recovered the context
        let needs_new_context = gpu_context
            .borrow()
            .as_ref()
            .is_none_or(|ctx| ctx.device_lost());

        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;

        let surface = if needs_new_context {
            log::warn!("GPU device lost, recreating context...");

            // Drop old resources to release Arc<Device>/Arc<Queue> and GPU resources
            self.resources = None;
            *gpu_context.borrow_mut() = None;

            // Wait briefly for the GPU driver to stabilize, then try to
            // recreate the context without software renderers. If this fails
            // the caller should request another frame and retry — the real GPU
            // may need more time to come back (e.g. after suspend/resume).
            std::thread::sleep(std::time::Duration::from_millis(350));

            let instance = WgpuContext::instance(Box::new(window.clone()));
            let surface = create_surface(&instance, window_handle.as_raw())?;
            let new_context =
                WgpuContext::new_rejecting_software(instance, &surface, self.compositor_gpu)?;
            *gpu_context.borrow_mut() = Some(new_context);
            surface
        } else {
            let ctx_ref = gpu_context.borrow();
            let instance = &ctx_ref.as_ref().unwrap().instance;
            create_surface(instance, window_handle.as_raw())?
        };

        let config = WgpuSurfaceConfig {
            size: gpui::Size {
                width: gpui::DevicePixels(self.surface_config.width as i32),
                height: gpui::DevicePixels(self.surface_config.height as i32),
            },
            transparent: self.surface_config.alpha_mode != wgpu::CompositeAlphaMode::Opaque,
            preferred_present_mode: Some(self.surface_config.present_mode),
        };
        let gpu_context = Rc::clone(gpu_context);
        let ctx_ref = gpu_context.borrow();
        let context = ctx_ref.as_ref().expect("context should exist");

        self.resources = None;
        self.atlas.handle_device_lost(context);

        *self = Self::new_internal(
            Some(gpu_context.clone()),
            context,
            surface,
            config,
            self.compositor_gpu,
            self.atlas.clone(),
        )?;

        log::info!("GPU recovery complete");
        Ok(())
    }
}

fn instance_range(range: Range<usize>) -> Range<u32> {
    range.start as u32..range.end as u32
}

#[cfg(not(target_family = "wasm"))]
fn create_surface(
    instance: &wgpu::Instance,
    raw_window_handle: raw_window_handle::RawWindowHandle,
) -> anyhow::Result<wgpu::Surface<'static>> {
    unsafe {
        instance
            .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                // Fall back to the display handle already provided via InstanceDescriptor::display.
                raw_display_handle: None,
                raw_window_handle,
            })
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

#[cfg(test)]
mod external_frame_tests {
    #[cfg(target_os = "linux")]
    use super::ExternalFrameSlot;
    use super::{ExternalFrameAcquisition, ExternalFrameOutcome, ExternalFrameState};
    #[cfg(target_os = "linux")]
    use super::{
        ExternalOwnership, VulkanExternalFrame, VulkanExternalSync, classify_external_sync_error,
    };
    #[cfg(target_os = "linux")]
    use ash::vk;
    #[cfg(target_os = "linux")]
    use std::sync::Arc;

    #[test]
    fn external_frame_outcomes_are_classified_strictly() {
        assert_eq!(
            ExternalFrameOutcome::classify(
                wgpu::Backend::Vulkan,
                ExternalFrameAcquisition::Prepared,
            ),
            ExternalFrameOutcome::Accepted
        );
        assert_eq!(
            ExternalFrameOutcome::classify(
                wgpu::Backend::Vulkan,
                ExternalFrameAcquisition::TransientFailure,
            ),
            ExternalFrameOutcome::TransientFailure
        );
        assert_eq!(
            ExternalFrameOutcome::classify(
                wgpu::Backend::Vulkan,
                ExternalFrameAcquisition::FatalFailure,
            ),
            ExternalFrameOutcome::FatalFailure
        );
        assert_eq!(
            ExternalFrameOutcome::classify(wgpu::Backend::Gl, ExternalFrameAcquisition::Prepared),
            ExternalFrameOutcome::Unsupported
        );
    }

    #[test]
    fn nv12_plane_views_use_explicit_aspect_formats() {
        let plane0 = match super::nv12_plane_view_descriptor(wgpu::TextureAspect::Plane0) {
            Some(descriptor) => descriptor,
            None => panic!("plane 0 descriptor must exist"),
        };
        assert_eq!(plane0.aspect, wgpu::TextureAspect::Plane0);
        assert_eq!(plane0.format, Some(wgpu::TextureFormat::R8Unorm));

        let plane1 = match super::nv12_plane_view_descriptor(wgpu::TextureAspect::Plane1) {
            Some(descriptor) => descriptor,
            None => panic!("plane 1 descriptor must exist"),
        };
        assert_eq!(plane1.aspect, wgpu::TextureAspect::Plane1);
        assert_eq!(plane1.format, Some(wgpu::TextureFormat::Rg8Unorm));
        assert!(super::nv12_plane_view_descriptor(wgpu::TextureAspect::All).is_none());
    }

    #[test]
    fn external_frame_state_keeps_only_the_latest_prepared_frame() {
        let mut state = ExternalFrameState::default();
        assert_eq!(
            state.stage(
                wgpu::Backend::Vulkan,
                ExternalFrameAcquisition::Prepared,
                Some(1),
            ),
            ExternalFrameOutcome::Accepted
        );
        assert_eq!(
            state.stage(
                wgpu::Backend::Vulkan,
                ExternalFrameAcquisition::Prepared,
                Some(2),
            ),
            ExternalFrameOutcome::Accepted
        );

        state.commit_after_submission();

        assert_eq!(state.displayed(), Some(&2));
        assert_eq!(state.latest(), None);
    }

    #[test]
    fn external_frame_submission_and_failure_preserve_previous_texture() {
        let mut state = ExternalFrameState::default();
        state.stage(
            wgpu::Backend::Vulkan,
            ExternalFrameAcquisition::Prepared,
            Some(1),
        );
        state.commit_after_submission();
        state.stage(
            wgpu::Backend::Vulkan,
            ExternalFrameAcquisition::Prepared,
            Some(2),
        );

        assert_eq!(
            state.stage(
                wgpu::Backend::Vulkan,
                ExternalFrameAcquisition::TransientFailure,
                None,
            ),
            ExternalFrameOutcome::TransientFailure
        );
        state.commit_after_submission();
        assert_eq!(state.displayed(), Some(&1));

        state.stage(
            wgpu::Backend::Vulkan,
            ExternalFrameAcquisition::Prepared,
            Some(2),
        );
        assert_eq!(
            state.stage(
                wgpu::Backend::Vulkan,
                ExternalFrameAcquisition::FatalFailure,
                None,
            ),
            ExternalFrameOutcome::FatalFailure
        );
        state.commit_after_submission();

        assert_eq!(state.displayed(), Some(&1));
        assert_eq!(state.latest(), None);
    }

    #[test]
    fn unsubmitted_external_frame_does_not_replace_previous_texture() {
        let mut state = ExternalFrameState::default();
        state.stage(
            wgpu::Backend::Vulkan,
            ExternalFrameAcquisition::Prepared,
            Some(1),
        );
        state.commit_after_submission();
        state.stage(
            wgpu::Backend::Vulkan,
            ExternalFrameAcquisition::Prepared,
            Some(2),
        );

        assert_eq!(state.displayed(), Some(&1));
    }

    #[test]
    fn clear_removes_displayed_and_never_submitted_latest_frames() {
        let mut state = ExternalFrameState::default();
        state.latest = Some(2);
        state.displayed = Some(1);

        state.clear();

        assert_eq!(state.latest(), None);
        assert_eq!(state.displayed(), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn matching_selection_prefers_latest_and_leaves_unrelated_latest_pending() {
        let mut state = ExternalFrameState::default();
        state.latest = Some(2);
        state.displayed = Some(1);

        assert_eq!(
            state.select_matching(|frame| *frame == 1),
            Some(ExternalFrameSlot::Displayed)
        );
        assert_eq!(
            state.select_matching(|frame| *frame == 2),
            Some(ExternalFrameSlot::Latest)
        );
        assert_eq!(state.select_matching(|frame| *frame == 3), None);
        assert_eq!(state.latest(), Some(&2));
        assert_eq!(state.displayed(), Some(&1));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn unmatched_scene_texture_schedules_no_external_barriers() {
        let mut state = ExternalFrameState::default();
        state.latest = Some(2);
        state.displayed = Some(1);

        assert_eq!(state.select_matching(|_| false), None);
        assert_eq!(state.latest(), Some(&2));
        assert_eq!(state.displayed(), Some(&1));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn failed_latest_submission_preserves_matching_displayed_frame() {
        let mut state = ExternalFrameState::default();
        state.stage(
            wgpu::Backend::Vulkan,
            ExternalFrameAcquisition::Prepared,
            Some(1),
        );
        state.commit_after_submission();
        state.stage(
            wgpu::Backend::Vulkan,
            ExternalFrameAcquisition::Prepared,
            Some(2),
        );
        state.stage(
            wgpu::Backend::Vulkan,
            ExternalFrameAcquisition::FatalFailure,
            None,
        );

        assert_eq!(
            state.select_matching(|frame| *frame == 1),
            Some(ExternalFrameSlot::Displayed)
        );
        assert_eq!(state.latest(), None);
        assert_eq!(state.displayed(), Some(&1));
    }

    #[test]
    fn unsupported_backend_does_not_change_external_frame_state() {
        let mut state = ExternalFrameState::default();
        state.stage(
            wgpu::Backend::Vulkan,
            ExternalFrameAcquisition::Prepared,
            Some(1),
        );

        for acquisition in [
            ExternalFrameAcquisition::Prepared,
            ExternalFrameAcquisition::TransientFailure,
            ExternalFrameAcquisition::FatalFailure,
        ] {
            assert_eq!(
                state.stage(wgpu::Backend::Gl, acquisition, Some(2)),
                ExternalFrameOutcome::Unsupported
            );
            assert_eq!(state.latest(), Some(&1));
            assert_eq!(state.displayed(), None);
        }

        state.commit_after_submission();
        assert_eq!(state.displayed(), Some(&1));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn ownership_family_mapping_is_restricted_to_external_families() {
        assert_eq!(
            super::queue_family(ExternalOwnership::External),
            vk::QUEUE_FAMILY_EXTERNAL
        );
        assert_eq!(
            super::queue_family(ExternalOwnership::Foreign),
            vk::QUEUE_FAMILY_FOREIGN_EXT
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn sync_file_import_uses_temporary_sync_fd_payload() {
        let import_info = VulkanExternalSync::sync_file_import_info(vk::Semaphore::null(), 17);

        assert_eq!(import_info.flags, vk::SemaphoreImportFlags::TEMPORARY);
        assert_eq!(
            import_info.handle_type,
            vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD
        );
        assert_eq!(import_info.fd, 17);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn sync_file_failures_preserve_recoverable_classification() {
        assert_eq!(
            classify_external_sync_error(vk::Result::ERROR_INVALID_EXTERNAL_HANDLE),
            ExternalFrameOutcome::TransientFailure
        );
        assert_eq!(
            classify_external_sync_error(vk::Result::ERROR_DEVICE_LOST),
            ExternalFrameOutcome::FatalFailure
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn uninitialized_external_state_is_rejected() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let released = Arc::new(AtomicBool::new(false));
        let released_for_lease = Arc::clone(&released);
        let sync_file = match std::fs::File::open("/dev/null") {
            Ok(file) => file.into(),
            Err(error) => panic!("/dev/null should be available: {error}"),
        };
        // SAFETY: This test intentionally supplies an invalid initial state to
        // verify constructor validation before the raw image is ever imported.
        let result = unsafe {
            super::VulkanExternalFrame::new(
                vk::Image::null(),
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUses::UNINITIALIZED,
                ExternalOwnership::External,
                sync_file,
                move || released_for_lease.store(true, Ordering::Release),
            )
        };
        let error = match result {
            Ok(_) => panic!("UNINITIALIZED must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("initialized producer contents"));
        assert!(released.load(Ordering::Acquire));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn unprepared_vulkan_frame_releases_its_lease() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let released = Arc::new(AtomicBool::new(false));
        let released_for_lease = Arc::clone(&released);
        let sync_file = match std::fs::File::open("/dev/null") {
            Ok(file) => file.into(),
            Err(error) => panic!("/dev/null should be available: {error}"),
        };
        // SAFETY: This test drops the descriptor before importing it, so no raw
        // image is dereferenced and the lease is the only observable invariant.
        let frame = match unsafe {
            VulkanExternalFrame::new(
                vk::Image::null(),
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUses::RESOURCE,
                ExternalOwnership::Foreign,
                sync_file,
                move || released_for_lease.store(true, Ordering::Release),
            )
        } {
            Ok(frame) => frame,
            Err(error) => panic!("valid initialized frame should construct: {error}"),
        };

        drop(frame);
        assert!(released.load(Ordering::Acquire));
    }
}

#[cfg(all(test, target_os = "linux"))]
mod vulkan_external_frame_integration {
    use super::{
        EXTERNAL_SEMAPHORE_FD_EXTENSION, ExternalFrameState, ExternalOwnership,
        PendingExternalFrame, PreparedExternalFrame, VulkanExternalSync, WgpuContext,
        classify_external_sync_error,
    };
    use ash::{khr::external_semaphore_fd, vk};
    use std::os::fd::{FromRawFd, OwnedFd};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    struct ProducerResources {
        device: ash::Device,
        signal_semaphore: Option<vk::Semaphore>,
        command_pool: Option<vk::CommandPool>,
    }

    impl ProducerResources {
        fn new(device: ash::Device, command_pool: vk::CommandPool) -> Self {
            Self {
                device,
                signal_semaphore: None,
                command_pool: Some(command_pool),
            }
        }

        fn destroy_handles(&mut self) {
            if let Some(signal_semaphore) = self.signal_semaphore.take() {
                // SAFETY: The caller waits until both the raw producer and normal consumer
                // submissions are complete before destroying the semaphore.
                unsafe {
                    self.device.destroy_semaphore(signal_semaphore, None);
                }
            }
            if let Some(command_pool) = self.command_pool.take() {
                // SAFETY: The caller waits until the raw producer command buffer is complete
                // before destroying the command pool that owns it.
                unsafe {
                    self.device.destroy_command_pool(command_pool, None);
                }
            }
        }
    }

    impl Drop for ProducerResources {
        fn drop(&mut self) {
            if self.signal_semaphore.is_none() && self.command_pool.is_none() {
                return;
            }
            // SAFETY: This test-only cleanup guard is used only after all Vulkan handles were
            // created on `device`; waiting for idle prevents an early destroy on error paths.
            if let Err(error) = unsafe { self.device.device_wait_idle() } {
                eprintln!("Vulkan producer cleanup wait failed: {error:?}");
            }
            self.destroy_handles();
        }
    }

    struct ProducerImageResources {
        device: ash::Device,
        image: vk::Image,
        memory: vk::DeviceMemory,
        handed_to_hal: bool,
    }

    impl ProducerImageResources {
        fn new(device: ash::Device, image: vk::Image, memory: vk::DeviceMemory) -> Self {
            Self {
                device,
                image,
                memory,
                handed_to_hal: false,
            }
        }

        fn hand_to_hal(&mut self) -> (vk::Image, vk::DeviceMemory) {
            self.handed_to_hal = true;
            (self.image, self.memory)
        }
    }

    impl Drop for ProducerImageResources {
        fn drop(&mut self) {
            if self.handed_to_hal {
                return;
            }
            // SAFETY: This test-only cleanup guard waits for any producer work before releasing
            // the image and its bound memory on an early-return path.
            if let Err(error) = unsafe { self.device.device_wait_idle() } {
                eprintln!("Vulkan image cleanup wait failed: {error:?}");
            }
            // SAFETY: The image and memory were created by `device` and are not owned by HAL.
            unsafe {
                self.device.destroy_image(self.image, None);
                self.device.free_memory(self.memory, None);
            }
        }
    }

    /// Run with `cargo test -p gpui_wgpu vulkan_external_frame_integration -- --nocapture`.
    /// The test skips when the host has no Vulkan adapter or sync-fd support.
    #[test]
    fn sync_file_round_trip_uses_normal_wgpu_submits() -> anyhow::Result<()> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let adapter = match gpui::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })) {
            Ok(adapter) => adapter,
            Err(error) => {
                eprintln!("SKIP Vulkan external-frame integration: no Vulkan adapter: {error}");
                return Ok(());
            }
        };
        let required_limits = wgpu::Limits::downlevel_defaults()
            .using_resolution(adapter.limits())
            .using_alignment(adapter.limits());
        let Some((device, queue)) = WgpuContext::create_vulkan_device_with_external_sync(
            &adapter,
            wgpu::Features::empty(),
            &required_limits,
        )?
        else {
            eprintln!(
                "SKIP Vulkan external-frame integration: {EXTERNAL_SEMAPHORE_FD_EXTENSION:?} unsupported"
            );
            return Ok(());
        };

        // SAFETY: The device was created by the Vulkan HAL path above.
        let Some(hal_device) = (unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }) else {
            eprintln!("SKIP Vulkan external-frame integration: Vulkan HAL device unavailable");
            return Ok(());
        };
        let raw_device = hal_device.raw_device().clone();
        let external_semaphore_fd = external_semaphore_fd::Device::new(
            hal_device.shared_instance().raw_instance(),
            &raw_device,
        );

        let raw_queue = hal_device.raw_queue();
        let queue_family = hal_device.queue_family_index();
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family);
        // SAFETY: `raw_device` owns the queue family used by `raw_queue`.
        let command_pool = unsafe { raw_device.create_command_pool(&command_pool_info, None) }?;
        let mut producer_resources = ProducerResources::new(raw_device.clone(), command_pool);
        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `command_pool` was created for the live producer queue family.
        let producer_command_buffer = unsafe {
            raw_device
                .allocate_command_buffers(&command_buffer_info)?
                .first()
                .copied()
                .ok_or_else(|| anyhow::anyhow!("Vulkan returned no producer command buffer"))?
        };

        let mut export_info = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut export_info);
        // SAFETY: `raw_device` is the live Vulkan device created above and the create info
        // requests a binary semaphore with the advertised sync-fd export handle type.
        let signal_semaphore = unsafe { raw_device.create_semaphore(&semaphore_info, None) }?;
        producer_resources.signal_semaphore = Some(signal_semaphore);

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let completion_probe = Arc::new(AtomicBool::new(false));
        let fallback_drop_probe = Arc::new(AtomicBool::new(false));
        {
            let texture_descriptor = wgpu::TextureDescriptor {
                label: Some("vulkan_external_frame_integration"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            };
            let hal_texture_descriptor = wgpu::hal::TextureDescriptor {
                label: Some("vulkan_external_frame_integration"),
                size: texture_descriptor.size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: texture_descriptor.format,
                usage: wgpu::TextureUses::RESOURCE,
                memory_flags: wgpu::hal::MemoryFlags::empty(),
                view_formats: Vec::new(),
            };
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            // SAFETY: `raw_device` is the live Vulkan device used by the wgpu HAL device.
            let external_image = unsafe { raw_device.create_image(&image_info, None) }?;
            let memory_requirements =
                unsafe { raw_device.get_image_memory_requirements(external_image) };
            let memory_properties = unsafe {
                hal_device
                    .shared_instance()
                    .raw_instance()
                    .get_physical_device_memory_properties(hal_device.raw_physical_device())
            };
            let Some(memory_type_index) = (0..memory_properties.memory_type_count).find(|index| {
                memory_requirements.memory_type_bits & (1 << index) != 0
                    && memory_properties
                        .memory_types
                        .get(*index as usize)
                        .is_some_and(|memory_type| {
                            memory_type
                                .property_flags
                                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                        })
            }) else {
                // SAFETY: The image was created by this device and has not been handed to wgpu.
                unsafe { raw_device.destroy_image(external_image, None) };
                anyhow::bail!("no device-local memory type supports the integration image");
            };
            let memory_info = vk::MemoryAllocateInfo::default()
                .allocation_size(memory_requirements.size)
                .memory_type_index(memory_type_index);
            // SAFETY: The allocation uses a memory type advertised for this image.
            let external_memory = match unsafe { raw_device.allocate_memory(&memory_info, None) } {
                Ok(memory) => memory,
                Err(error) => {
                    // SAFETY: The image was created by this device and has not been handed to
                    // wgpu because its memory allocation failed.
                    unsafe { raw_device.destroy_image(external_image, None) };
                    return Err(error.into());
                }
            };
            // SAFETY: The image and memory were created by the same Vulkan device and the
            // allocation satisfies the image's memory requirements.
            if let Err(error) =
                unsafe { raw_device.bind_image_memory(external_image, external_memory, 0) }
            {
                // SAFETY: Neither handle has been handed to wgpu because binding failed.
                unsafe {
                    raw_device.free_memory(external_memory, None);
                    raw_device.destroy_image(external_image, None);
                }
                return Err(error.into());
            }
            let mut image_resources =
                ProducerImageResources::new(raw_device.clone(), external_image, external_memory);

            // SAFETY: The image is bound and the command buffer is recording on its owning queue.
            unsafe {
                raw_device.begin_command_buffer(
                    producer_command_buffer,
                    &vk::CommandBufferBeginInfo::default(),
                )?;
                let acquire_barrier = vk::ImageMemoryBarrier::default()
                    .image(external_image)
                    .subresource_range(subresource_range)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
                raw_device.cmd_pipeline_barrier(
                    producer_command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[acquire_barrier],
                );
                raw_device.cmd_clear_color_image(
                    producer_command_buffer,
                    external_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue {
                        float32: [1.0, 0.0, 0.0, 1.0],
                    },
                    &[subresource_range],
                );
                let release_barrier = vk::ImageMemoryBarrier::default()
                    .image(external_image)
                    .subresource_range(subresource_range)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::empty())
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(queue_family)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL);
                raw_device.cmd_pipeline_barrier(
                    producer_command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[release_barrier],
                );
                raw_device.end_command_buffer(producer_command_buffer)?;
                let command_buffers = [producer_command_buffer];
                let signal_semaphores = [signal_semaphore];
                let submit_info = vk::SubmitInfo::default()
                    .command_buffers(&command_buffers)
                    .signal_semaphores(&signal_semaphores);
                raw_device.queue_submit(raw_queue, &[submit_info], vk::Fence::null())?;
            }

            let get_fd_info = vk::SemaphoreGetFdInfoKHR::default()
                .semaphore(signal_semaphore)
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
            // SAFETY: The extension loader and export semaphore belong to `raw_device`.
            let sync_fd = unsafe { external_semaphore_fd.get_semaphore_fd(&get_fd_info) }?;
            // SAFETY: Vulkan returned ownership of this valid descriptor to the caller.
            let sync_file = unsafe { OwnedFd::from_raw_fd(sync_fd) };

            let external_sync = VulkanExternalSync {
                fd: Some(sync_file),
                device: raw_device.clone(),
                external_semaphore_fd: external_semaphore_fd::Device::new(
                    hal_device.shared_instance().raw_instance(),
                    &raw_device,
                ),
                semaphore: None,
                completion_probe: Some(Arc::clone(&completion_probe)),
            };

            let lease_probe = Arc::clone(&fallback_drop_probe);
            let image_device = raw_device.clone();
            let (external_image, external_memory) = image_resources.hand_to_hal();
            // SAFETY: The image is owned by this test until wgpu releases the imported texture;
            // the callback then destroys the exact image handle created above.
            let hal_texture = unsafe {
                hal_device.texture_from_raw(
                    external_image,
                    &hal_texture_descriptor,
                    Some(Box::new(move || {
                        lease_probe.store(true, Ordering::Release);
                        // SAFETY: wgpu has finished using the image before invoking this drop
                        // callback, and this callback owns the image handle.
                        image_device.destroy_image(external_image, None);
                        image_device.free_memory(external_memory, None);
                    })),
                    wgpu::hal::vulkan::TextureMemory::External,
                )
            };
            // SAFETY: `hal_texture` was created by this device and remains alive in `texture`.
            let texture = unsafe {
                device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
                    hal_texture,
                    &texture_descriptor,
                    wgpu::TextureUses::RESOURCE,
                )
            };
            let mut prepared = PreparedExternalFrame {
                external_sync: Some(external_sync),
                initial_state: wgpu::TextureUses::RESOURCE,
                ownership: Some(ExternalOwnership::External),
                texture: Arc::new(texture),
            };

            let external_view = prepared
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vulkan_external_frame_integration_sample_result"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vulkan_external_frame_integration_sample"),
                source: wgpu::ShaderSource::Wgsl(
                    "@group(0) @binding(0) var input_texture: texture_2d<f32>;\n\
                     @group(0) @binding(1) var<storage, read_write> result: array<vec4<f32>>;\n\
                     @compute @workgroup_size(1)\n\
                     fn main() {\n\
                         result[0] = textureLoad(input_texture, vec2<i32>(0, 0), 0);\n\
                     }"
                    .into(),
                ),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("vulkan_external_frame_integration_sample"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vulkan_external_frame_integration_sample"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&external_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: result_buffer.as_entire_binding(),
                    },
                ],
            });

            // Keep ownership transitions separate from the normal consumer encoder. All three
            // command buffers are submitted together through the normal wgpu queue.
            let mut acquire_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vulkan_external_frame_integration_acquire"),
                });
            let mut consumer_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vulkan_external_frame_integration_consumer"),
                });
            let mut release_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vulkan_external_frame_integration_release"),
                });
            assert_eq!(
                prepared.encode_acquire_ownership(&mut acquire_encoder),
                Ok(()),
                "Vulkan ownership acquire must use the live external texture"
            );

            {
                let mut pass = consumer_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vulkan_external_frame_integration_consumer"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            assert_eq!(
                prepared.encode_release_ownership(&mut release_encoder),
                Ok(()),
                "Vulkan ownership release must use the live external texture"
            );

            let acquire_command_buffer = acquire_encoder.finish();
            let consumer_command_buffer = consumer_encoder.finish();
            let release_command_buffer = release_encoder.finish();
            prepared
                .import_sync(&queue)
                .map_err(|error| anyhow::anyhow!("importing exported sync fd: {error:?}"))?;
            queue.submit([
                acquire_command_buffer,
                consumer_command_buffer,
                release_command_buffer,
            ]);
            prepared.on_submitted(&queue);
            assert!(prepared.external_sync.is_none());

            device.poll(wgpu::PollType::wait_indefinitely())?;
            assert!(completion_probe.load(Ordering::Acquire));

            let mut held_acquire_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vulkan_external_frame_integration_held_acquire"),
                });
            let mut held_consumer_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vulkan_external_frame_integration_held_consumer"),
                });
            let mut held_release_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vulkan_external_frame_integration_held_release"),
                });
            assert_eq!(
                prepared.encode_acquire_ownership(&mut held_acquire_encoder),
                Ok(()),
                "held Vulkan redraw must reacquire ownership without a new sync file"
            );
            {
                let mut pass =
                    held_consumer_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("vulkan_external_frame_integration_held_consumer"),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            assert_eq!(
                prepared.encode_release_ownership(&mut held_release_encoder),
                Ok(()),
                "held Vulkan redraw must release ownership after sampling"
            );
            queue.submit([
                held_acquire_encoder.finish(),
                held_consumer_encoder.finish(),
                held_release_encoder.finish(),
            ]);
            prepared.on_submitted(&queue);
            // Model clearing the displayed slot after submission. The completion callback keeps
            // the texture and producer lease alive until the queued redraw finishes.
            let mut displayed_state = ExternalFrameState::default();
            displayed_state.displayed = Some(PendingExternalFrame::Prepared(prepared));
            displayed_state.clear();
            assert!(!fallback_drop_probe.load(Ordering::Acquire));
        }

        let submission_completed = Arc::new(AtomicBool::new(false));
        let completed_for_callback = Arc::clone(&submission_completed);
        queue.on_submitted_work_done(move || {
            completed_for_callback.store(true, Ordering::Release);
        });
        device.poll(wgpu::PollType::wait_indefinitely())?;
        assert!(completion_probe.load(Ordering::Acquire));
        assert!(submission_completed.load(Ordering::Acquire));
        assert!(fallback_drop_probe.load(Ordering::Acquire));
        producer_resources.destroy_handles();

        let invalid_file = std::fs::File::open("/dev/null")?;
        let mut invalid_sync = VulkanExternalSync {
            fd: Some(invalid_file.into()),
            device: raw_device.clone(),
            external_semaphore_fd: external_semaphore_fd::Device::new(
                hal_device.shared_instance().raw_instance(),
                &raw_device,
            ),
            semaphore: None,
            completion_probe: None,
        };
        let invalid_error = {
            // SAFETY: The queue is the Vulkan queue created with `raw_device`.
            let Some(hal_queue) = (unsafe { queue.as_hal::<wgpu::hal::api::Vulkan>() }) else {
                eprintln!("SKIP Vulkan external-frame integration: Vulkan HAL queue unavailable");
                return Ok(());
            };
            match invalid_sync.import(&hal_queue) {
                Ok(()) => panic!("/dev/null must not import as a sync file"),
                Err(error) => error,
            }
        };
        assert_eq!(
            classify_external_sync_error(invalid_error),
            super::ExternalFrameOutcome::TransientFailure
        );
        Ok(())
    }
}

struct RenderingParameters {
    path_sample_count: u32,
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
}

impl RenderingParameters {
    fn new(adapter: &wgpu::Adapter, surface_format: wgpu::TextureFormat) -> Self {
        use std::env;

        let format_features = adapter.get_texture_format_features(surface_format);
        let path_sample_count = [4, 2, 1]
            .into_iter()
            .find(|&n| format_features.flags.sample_count_supported(n))
            .unwrap_or(1);

        let gamma = env::var("ZED_FONTS_GAMMA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.8_f32)
            .clamp(1.0, 2.2);
        let gamma_ratios = get_gamma_correction_ratios(gamma);

        let grayscale_enhanced_contrast = env::var("ZED_FONTS_GRAYSCALE_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0_f32)
            .max(0.0);

        let subpixel_enhanced_contrast = env::var("ZED_FONTS_SUBPIXEL_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5_f32)
            .max(0.0);

        Self {
            path_sample_count,
            gamma_ratios,
            grayscale_enhanced_contrast,
            subpixel_enhanced_contrast,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{MonochromeSprite, PolychromeSprite, Quad, Shadow, SubpixelSprite, Underline};

    #[test]
    fn webgl_shader_is_valid_wgsl_without_storage_buffers() {
        assert!(!WEBGL_SHADERS.contains("var<storage"));
        validate_wgsl(WEBGL_SHADERS, naga::valid::Capabilities::empty());
    }

    #[test]
    fn storage_buffer_shader_is_valid_wgsl() {
        validate_wgsl(STORAGE_BUFFER_SHADERS, naga::valid::Capabilities::empty());
    }

    #[test]
    fn subpixel_shader_is_valid_wgsl() {
        validate_wgsl(
            SUBPIXEL_SHADERS,
            naga::valid::Capabilities::DUAL_SOURCE_BLENDING,
        );
    }

    fn validate_wgsl(source: &str, capabilities: naga::valid::Capabilities) {
        let module = naga::front::wgsl::parse_str(source).expect("shader should parse");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
            .validate(&module)
            .expect("shader should validate");
    }

    #[test]
    fn webgl_record_sizes_match_shader_word_strides() {
        assert_eq!(std::mem::size_of::<Quad>(), 40 * 4);
        assert_eq!(std::mem::size_of::<Shadow>(), 28 * 4);
        assert_eq!(std::mem::size_of::<PathRasterizationVertex>(), 26 * 4);
        assert_eq!(std::mem::size_of::<PathSprite>(), 4 * 4);
        assert_eq!(std::mem::size_of::<Underline>(), 16 * 4);
        assert_eq!(std::mem::size_of::<MonochromeSprite>(), 28 * 4);
        assert_eq!(std::mem::size_of::<SubpixelSprite>(), 28 * 4);
        assert_eq!(std::mem::size_of::<PolychromeSprite>(), 24 * 4);
    }

    #[test]
    fn nv12_tuple_apis_are_available_without_gpu_resources() {
        fn assert_into_surface_source<T: Into<gpui::SurfaceSource>>() {}

        assert_into_surface_source::<(
            Arc<wgpu::Texture>,
            Arc<wgpu::Texture>,
            gpui::Size<gpui::DevicePixels>,
        )>();
        assert_into_surface_source::<(
            Arc<wgpu::Texture>,
            Arc<wgpu::Texture>,
            gpui::Size<gpui::DevicePixels>,
            gpui::Nv12ColorTransform,
        )>();
    }

    #[test]
    fn nv12_params_preserve_default_and_column_major_matrix_layout() {
        assert_eq!(
            gpui::Nv12ColorTransform::default().yuv_to_rgb,
            [
                [1.0000, 1.0000, 1.0000, 0.0],
                [0.0000, -0.3441, 1.7720, 0.0],
                [1.4020, -0.7141, 0.0000, 0.0],
                [-0.7010, 0.5291, -0.8860, 1.0],
            ]
        );

        let transform = gpui::Nv12ColorTransform {
            yuv_to_rgb: [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
            ],
        };
        let params = SurfaceParams::new(
            PodBounds {
                origin: [0.0; 2],
                size: [0.0; 2],
            },
            PodBounds {
                origin: [0.0; 2],
                size: [0.0; 2],
            },
            transform,
        );

        assert_eq!(params.yuv_to_rgb, transform.yuv_to_rgb);
        assert_eq!(
            bytemuck::bytes_of(&params).get(32..96),
            Some(bytemuck::bytes_of(&transform.yuv_to_rgb))
        );
        assert!(
            include_str!("shaders.wgsl").contains("return surface_locals.yuv_to_rgb * y_cb_cr;")
        );
    }
}
