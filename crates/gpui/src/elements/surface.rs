use crate::{
    Bounds, Element, ElementId, GlobalElementId, InspectorElementId, InteractiveElement,
    Interactivity, IntoElement, LayoutId, ObjectFit, Pixels, Style, StyleRefinement, Styled,
    Window,
};
#[cfg(feature = "wgpu")]
use crate::{DevicePixels, Size};
#[cfg(target_os = "macos")]
use core_video::pixel_buffer::CVPixelBuffer;
use refineable::Refineable;
#[cfg(feature = "wgpu")]
use std::sync::Arc;

/// Color conversion applied to sampled NV12 values.
///
/// The input vector is ordered as `[Y, Cb, Cr, 1]`, matching the values
/// sampled by the NV12 shader. `yuv_to_rgb[column][row]` stores one column of
/// the matrix, which is the column-major layout expected by WGSL's
/// `mat4x4<f32>` and its `matrix * vector` multiplication. The fourth column
/// therefore contains the additive offsets, including the `-0.5` chroma
/// centering used by the default conversion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nv12ColorTransform {
    /// Column-major YUV-to-RGB matrix for an input `[Y, Cb, Cr, 1]` vector.
    pub yuv_to_rgb: [[f32; 4]; 4],
    /// Encoded transfer function applied after the matrix conversion.
    pub transfer: crate::VideoTransferFunction,
}

impl Default for Nv12ColorTransform {
    fn default() -> Self {
        Self {
            transfer: crate::VideoTransferFunction::Srgb,
            yuv_to_rgb: [
                [1.0000, 1.0000, 1.0000, 0.0],
                [0.0000, -0.3441, 1.7720, 0.0],
                [1.4020, -0.7141, 0.0000, 0.0],
                [-0.7010, 0.5291, -0.8860, 1.0],
            ],
        }
    }
}

/// Source content for a [`Surface`] element.
#[derive(Clone)]
pub enum SurfaceSource {
    /// A macOS CoreVideo pixel buffer (zero-copy, no pre-registration needed).
    #[cfg(target_os = "macos")]
    Surface(CVPixelBuffer),
    /// A validated RGBA/BGRA texture with explicit alpha and color metadata.
    #[cfg(feature = "wgpu")]
    Rgba(crate::RgbaTextureSource),
    /// Validated NV12 planes with explicit color-conversion metadata.
    #[cfg(feature = "wgpu")]
    Nv12Texture {
        /// Luma plane.
        y_texture: Arc<wgpu::Texture>,
        /// Interleaved chroma plane.
        cb_cr_texture: Arc<wgpu::Texture>,
        /// Source dimensions for object-fit.
        native_size: Size<DevicePixels>,
    },
    /// Two-plane NV12 wgpu texture with an explicit color transform.
    #[cfg(feature = "wgpu")]
    #[expect(missing_docs)]
    Nv12TextureWithColorTransform {
        y_texture: Arc<wgpu::Texture>,
        cb_cr_texture: Arc<wgpu::Texture>,
        native_size: Size<DevicePixels>,
        color_transform: Nv12ColorTransform,
    },
    /// One multiplanar NV12 wgpu texture with an explicit color transform.
    #[cfg(feature = "wgpu")]
    #[expect(missing_docs)]
    Nv12MultiplanarTexture {
        texture: Arc<wgpu::Texture>,
        native_size: Size<DevicePixels>,
        color_transform: Nv12ColorTransform,
    },
    /// Validated NV12 planes with explicit color metadata.
    #[cfg(feature = "wgpu")]
    Nv12(crate::Nv12TextureSource),
}

#[cfg(target_os = "macos")]
impl From<CVPixelBuffer> for SurfaceSource {
    fn from(value: CVPixelBuffer) -> Self {
        SurfaceSource::Surface(value)
    }
}

#[cfg(feature = "wgpu")]
impl From<crate::RgbaTextureSource> for SurfaceSource {
    fn from(source: crate::RgbaTextureSource) -> Self {
        Self::Rgba(source)
    }
}

#[cfg(feature = "wgpu")]
impl From<crate::Nv12TextureSource> for SurfaceSource {
    fn from(source: crate::Nv12TextureSource) -> Self {
        Self::Nv12(source)
    }
}

#[cfg(feature = "wgpu")]
impl
    From<(
        Arc<wgpu::Texture>,
        Arc<wgpu::Texture>,
        Size<DevicePixels>,
        Nv12ColorTransform,
    )> for SurfaceSource
{
    fn from(
        (y_texture, cb_cr_texture, native_size, color_transform): (
            Arc<wgpu::Texture>,
            Arc<wgpu::Texture>,
            Size<DevicePixels>,
            Nv12ColorTransform,
        ),
    ) -> Self {
        SurfaceSource::Nv12TextureWithColorTransform {
            y_texture,
            cb_cr_texture,
            native_size,
            color_transform,
        }
    }
}

#[cfg(feature = "wgpu")]
impl From<(Arc<wgpu::Texture>, Size<DevicePixels>, Nv12ColorTransform)> for SurfaceSource {
    fn from(
        (texture, native_size, color_transform): (
            Arc<wgpu::Texture>,
            Size<DevicePixels>,
            Nv12ColorTransform,
        ),
    ) -> Self {
        SurfaceSource::Nv12MultiplanarTexture {
            texture,
            native_size,
            color_transform,
        }
    }
}

/// A GPU texture composited into the UI.
///
/// # Examples
///
/// ```ignore
/// // Validated wgpu texture source with object-fit (cross-platform):
/// surface(rgba_source).object_fit(ObjectFit::Contain)
///
/// // macOS zero-copy via CoreVideo pixel buffer (Metal backend):
/// surface(pixel_buffer).object_fit(ObjectFit::Contain)
///
/// // 3D viewport with mouse input:
/// surface(rgba_source)
///     .object_fit(ObjectFit::Fill)
///     .on_scroll(cx.listener(|this, event, window, cx| { ... }))
/// ```
pub fn surface(source: impl Into<SurfaceSource>) -> Surface {
    let source = source.into();
    Surface {
        source,
        object_fit: ObjectFit::Contain,
        interactivity: Interactivity::new(),
        style: StyleRefinement::default(),
    }
}

/// A surface element.
pub struct Surface {
    source: SurfaceSource,
    object_fit: ObjectFit,
    interactivity: Interactivity,
    style: StyleRefinement,
}

impl Surface {
    /// Set the object fit for the surface.
    pub fn object_fit(mut self, object_fit: ObjectFit) -> Self {
        self.object_fit = object_fit;
        self
    }
}

impl Element for Surface {
    type RequestLayoutState = ();
    type PrepaintState = Option<crate::Hitbox>;

    fn id(&self) -> Option<ElementId> {
        self.interactivity.element_id.clone()
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        self.interactivity.source_location()
    }

    fn request_layout(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut crate::App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut style = Style::default();
        style.refine(&self.style);

        // Make the surface fill its parent so ObjectFit::Contain has a
        // non-zero layout bound to work with.  Without explicit size,
        // GPUI's layout assigns zero bounds to auto-sized children in
        // flex containers — see issue #1.
        let has_native_size = match &self.source {
            #[cfg(target_os = "macos")]
            SurfaceSource::Surface(pixel_buffer) => {
                let h = pixel_buffer.get_height();
                if h > 0 {
                    style.aspect_ratio = Some(pixel_buffer.get_width() as f32 / h as f32);
                    true
                } else {
                    false
                }
            }
            #[cfg(feature = "wgpu")]
            SurfaceSource::Rgba(source) => {
                let size = source.native_size();
                if size.height.0 > 0 {
                    style.aspect_ratio = Some(size.width.0 as f32 / size.height.0 as f32);
                    true
                } else {
                    false
                }
            }
            #[cfg(feature = "wgpu")]
            SurfaceSource::Nv12Texture { native_size, .. }
            | SurfaceSource::Nv12TextureWithColorTransform { native_size, .. }
            | SurfaceSource::Nv12MultiplanarTexture { native_size, .. } => {
                if native_size.height.0 > 0 {
                    style.aspect_ratio =
                        Some(native_size.width.0 as f32 / native_size.height.0 as f32);
                    true
                } else {
                    false
                }
            }
            #[cfg(feature = "wgpu")]
            SurfaceSource::Nv12(source) => {
                let size = source.native_size();
                if size.height.0 > 0 {
                    style.aspect_ratio = Some(size.width.0 as f32 / size.height.0 as f32);
                    true
                } else {
                    false
                }
            }
            #[allow(unreachable_patterns)]
            _ => false,
        };

        if has_native_size {
            style.size.width = crate::relative(1.0).into();
            style.size.height = crate::relative(1.0).into();
        }

        let layout_id = window.request_layout(style, [], cx);
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut crate::App,
    ) -> Self::PrepaintState {
        self.interactivity.prepaint(
            global_id,
            inspector_id,
            bounds,
            bounds.size,
            window,
            cx,
            |_, _, hitbox, _, _| hitbox,
        )
    }

    fn paint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        hitbox: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut crate::App,
    ) {
        self.interactivity.paint(
            _global_id,
            _inspector_id,
            bounds,
            hitbox.as_ref(),
            window,
            cx,
            |_, window, _| {
                match &self.source {
                    // macOS direct CVPixelBuffer path (zero-copy).
                    #[cfg(target_os = "macos")]
                    SurfaceSource::Surface(pixel_buffer) => {
                        let device_size = crate::size(
                            crate::DevicePixels::from(pixel_buffer.get_width() as i32),
                            crate::DevicePixels::from(pixel_buffer.get_height() as i32),
                        );
                        let paint_bounds = self.object_fit.get_bounds(bounds, device_size);
                        window.paint_surface(paint_bounds, pixel_buffer.clone());
                    }
                    #[cfg(feature = "wgpu")]
                    SurfaceSource::Rgba(source) => {
                        let paint_bounds = self.object_fit.get_bounds(bounds, source.native_size());
                        window.paint_surface_with_rgba_source(paint_bounds, source.clone());
                    }
                    #[cfg(feature = "wgpu")]
                    #[cfg(feature = "wgpu")]
            SurfaceSource::Nv12(source) => {
                        let paint_bounds = self.object_fit.get_bounds(bounds, source.native_size());
                        window.paint_surface_with_nv12_source(paint_bounds, source.clone());
                    }
                    #[cfg(feature = "wgpu")]
                    SurfaceSource::Nv12TextureWithColorTransform {
                        y_texture,
                        cb_cr_texture,
                        native_size,
                        color_transform,
                    } => {
                        let paint_bounds = self.object_fit.get_bounds(bounds, *native_size);
                        eprintln!(
                            "NV12 paint: layout_bounds={:?}×{:?}, native={:?}×{:?}, paint_bounds={:?}×{:?}",
                            bounds.origin, bounds.size,
                            native_size.width, native_size.height,
                            paint_bounds.origin, paint_bounds.size,
                        );
                        window.paint_surface_with_nv12_texture_with_color_transform(
                            paint_bounds,
                            y_texture.clone(),
                            cb_cr_texture.clone(),
                            *native_size,
                            *color_transform,
                        );
                    }
                    #[cfg(feature = "wgpu")]
                    SurfaceSource::Nv12MultiplanarTexture {
                        texture,
                        native_size,
                        color_transform,
                    } => {
                        let paint_bounds = self.object_fit.get_bounds(bounds, *native_size);
                        window.paint_surface_with_nv12_multiplanar_texture(
                            paint_bounds,
                            texture.clone(),
                            *native_size,
                            *color_transform,
                        );
                    }
                    #[allow(unreachable_patterns)]
                    _ => {}
                }
            },
        );
    }
}

impl IntoElement for Surface {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Styled for Surface {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

impl InteractiveElement for Surface {
    fn interactivity(&mut self) -> &mut Interactivity {
        &mut self.interactivity
    }
}

impl crate::StatefulInteractiveElement for Surface {}
