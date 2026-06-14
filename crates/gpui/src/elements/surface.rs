use crate::{
    Bounds, DevicePixels, Element, ElementId, GlobalElementId, InspectorElementId,
    InteractiveElement, Interactivity, IntoElement, LayoutId, ObjectFit, Pixels, Size, Style,
    StyleRefinement, Styled, Window,
};
#[cfg(target_os = "macos")]
use core_video::pixel_buffer::CVPixelBuffer;
#[cfg(feature = "wgpu")]
use std::sync::Arc;
use refineable::Refineable;

/// Source content for a [`Surface`] element.
#[derive(Clone)]
pub enum SurfaceSource {
    /// A macOS CoreVideo pixel buffer (zero-copy, no pre-registration needed).
    #[cfg(target_os = "macos")]
    Surface(CVPixelBuffer),
    /// A wgpu texture with its descriptor for [`ObjectFit`] sizing.
    #[cfg(feature = "wgpu")]
    Texture {
        texture: Arc<wgpu::Texture>,
        /// The texture's native size in device pixels, used for
        /// [`ObjectFit`] calculations. When `None`, the texture
        /// fills the layout bounds (ignoring aspect ratio).
        native_size: Option<Size<DevicePixels>>,
    },
}

#[cfg(target_os = "macos")]
impl From<CVPixelBuffer> for SurfaceSource {
    fn from(value: CVPixelBuffer) -> Self {
        SurfaceSource::Surface(value)
    }
}

#[cfg(feature = "wgpu")]
impl From<Arc<wgpu::Texture>> for SurfaceSource {
    fn from(texture: Arc<wgpu::Texture>) -> Self {
        SurfaceSource::Texture {
            texture,
            native_size: None,
        }
    }
}

#[cfg(feature = "wgpu")]
impl From<(Arc<wgpu::Texture>, crate::GpuTextureDescriptor)> for SurfaceSource {
    fn from((texture, descriptor): (Arc<wgpu::Texture>, crate::GpuTextureDescriptor)) -> Self {
        SurfaceSource::Texture {
            texture,
            native_size: Some(descriptor.size),
        }
    }
}

#[cfg(feature = "wgpu")]
impl From<(Arc<wgpu::Texture>, Option<Size<DevicePixels>>)> for SurfaceSource {
    fn from(
        (texture, native_size): (Arc<wgpu::Texture>, Option<Size<DevicePixels>>),
    ) -> Self {
        SurfaceSource::Texture {
            texture,
            native_size,
        }
    }
}

/// A GPU texture composited into the UI.
///
/// # Examples
///
/// ```ignore
/// // wgpu texture with object-fit (cross-platform):
/// surface((video_frame_texture, descriptor)).object_fit(ObjectFit::Contain)
///
/// // macOS zero-copy via CoreVideo pixel buffer (Metal backend):
/// surface(pixel_buffer).object_fit(ObjectFit::Contain)
///
/// // Without object-fit (fills bounds):
/// surface(video_frame_texture)
///
/// // 3D viewport with mouse input:
/// surface((render_target, descriptor))
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

        // Communicate the texture's native aspect ratio to the layout
        // engine. This allows the surface to size proportionally in flex
        // containers, matching the behavior of `img()`.
        match &self.source {
            #[cfg(target_os = "macos")]
            SurfaceSource::Surface(pixel_buffer) => {
                let h = pixel_buffer.get_height();
                if h > 0 {
                    style.aspect_ratio = Some(pixel_buffer.get_width() as f32 / h as f32);
                }
            }
            #[cfg(feature = "wgpu")]
            SurfaceSource::Texture {
                native_size: Some(size),
                ..
            } => {
                if size.height.0 > 0 {
                    style.aspect_ratio = Some(size.width.0 as f32 / size.height.0 as f32);
                }
            }
            _ => {}
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
                    // Cross-platform wgpu texture path.
                    #[cfg(feature = "wgpu")]
                    SurfaceSource::Texture {
                        texture,
                        native_size: Some(size),
                    } => {
                        let paint_bounds = self.object_fit.get_bounds(bounds, *size);
                        window.paint_surface_with_texture(paint_bounds, texture.clone());
                    }
                    #[cfg(feature = "wgpu")]
                    SurfaceSource::Texture {
                        texture,
                        native_size: None,
                    } => {
                        window.paint_surface_with_texture(bounds, texture.clone());
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
