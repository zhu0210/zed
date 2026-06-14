//! Demonstrates CVPixelBuffer compositing via `surface()` on macOS.
//!
//! Two paths are shown side-by-side:
//!
//! 1. **Direct**: `surface(pixel_buffer)` — zero-copy Metal path
//! 2. **wgpu import**: `import_cv_pixel_buffer_to_wgpu(&pb, &gpu)` →
//!    `surface((texture, desc))` — cross-platform wgpu path
//!
//! ```sh
//! cargo run -p gpui_macos --example cv_interop
//! ```

use core_foundation::{
    base::{CFType, TCFType},
    dictionary::CFDictionary,
    string::CFString,
};
use core_video::pixel_buffer::{
    CVPixelBuffer, kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferLock_ReadOnly,
    kCVPixelFormatType_32BGRA,
};
use gpui::{
    App, Application, Bounds, Context, DevicePixels, GpuTextureColorSpace, GpuTextureDescriptor,
    GpuTextureFormat, IntoElement, ObjectFit, ParentElement, Render, SharedString, Styled, Window,
    WindowBounds, WindowOptions, div, prelude::*, px, rgb, size, surface,
};
use gpui_macos::MacPlatform;
use std::sync::Arc;

const PB_WIDTH: u32 = 320;
const PB_HEIGHT: u32 = 240;

struct CvInteropDemo {
    label: SharedString,
    device_info: SharedString,
    test_pixel_buffer: Option<CVPixelBuffer>,
    imported_texture: Option<Arc<gpui_wgpu::wgpu::Texture>>,
}

/// Create a BGRA8 CVPixelBuffer with a four-quadrant test pattern:
/// top-left red, top-right green, bottom-left blue, bottom-right white.
fn create_test_pixel_buffer() -> CVPixelBuffer {
    let empty_iosurface_props: CFDictionary<CFString, CFType> =
        CFDictionary::from_CFType_pairs(&[]);

    let attrs: CFDictionary<CFString, CFType> = CFDictionary::from_CFType_pairs(&[(
        unsafe { CFString::wrap_under_get_rule(kCVPixelBufferIOSurfacePropertiesKey) },
        empty_iosurface_props.as_CFType(),
    )]);

    let pb = CVPixelBuffer::new(
        kCVPixelFormatType_32BGRA,
        PB_WIDTH as usize,
        PB_HEIGHT as usize,
        Some(&attrs),
    )
    .expect("failed to create CVPixelBuffer");
    pb.lock_base_address(kCVPixelBufferLock_ReadOnly);
    let base = unsafe { pb.get_base_address() };
    assert!(!base.is_null());

    let bytes_per_row = pb.get_bytes_per_row();
    let data = unsafe {
        std::slice::from_raw_parts_mut(base as *mut u8, bytes_per_row * PB_HEIGHT as usize)
    };

    for row in 0..PB_HEIGHT {
        for col in 0..PB_WIDTH {
            let idx = (row as usize * bytes_per_row) + (col as usize * 4);
            let (b, g, r) = match (col >= PB_WIDTH / 2, row >= PB_HEIGHT / 2) {
                (false, false) => (0, 0, 255),   // top-left:  red
                (true, false) => (0, 255, 0),    // top-right: green
                (false, true) => (255, 0, 0),    // bot-left:  blue
                (true, true) => (255, 255, 255), // bot-right: white
            };
            data[idx] = b;
            data[idx + 1] = g;
            data[idx + 2] = r;
            data[idx + 3] = 255;
        }
    }

    pb.unlock_base_address(kCVPixelBufferLock_ReadOnly);
    pb
}

fn make_descriptor(pb: &CVPixelBuffer) -> GpuTextureDescriptor {
    GpuTextureDescriptor {
        size: size(
            DevicePixels::from(pb.get_width() as i32),
            DevicePixels::from(pb.get_height() as i32),
        ),
        format: GpuTextureFormat::Bgra8Unorm,
        color_space: GpuTextureColorSpace::Srgb,
    }
}

impl Render for CvInteropDemo {
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        if self.test_pixel_buffer.is_none() {
            log::info!(
                "cv_interop: creating test CVPixelBuffer {}×{}",
                PB_WIDTH,
                PB_HEIGHT
            );
            self.test_pixel_buffer = Some(create_test_pixel_buffer());
        }

        if self.imported_texture.is_none() {
            if let Some(ref pb) = self.test_pixel_buffer {
                if let Some(gpu) = window.gpu_context() {
                    log::info!(
                        "cv_interop: importing CVPixelBuffer to wgpu. \
                         format={:?}, dual_src={}",
                        gpu.color_texture_format,
                        gpu.supports_dual_source_blending
                    );
                    self.device_info = SharedString::from(format!(
                        "wgpu: {:?} dual_src={}",
                        gpu.color_texture_format, gpu.supports_dual_source_blending
                    ));
                    self.imported_texture = gpui_macos::import_cv_pixel_buffer_to_wgpu(pb, &gpu);
                    if self.imported_texture.is_some() {
                        log::info!("cv_interop: import succeeded");
                    } else {
                        log::warn!(
                            "cv_interop: import returned None — CPU fallback may have been used"
                        );
                    }
                }
            }
        }

        let descriptor = self.test_pixel_buffer.as_ref().map(make_descriptor);

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(rgb(0x1e1e2e))
            .gap_4()
            .p_8()
            .child(
                div()
                    .text_xl()
                    .font_weight(gpui::FontWeight::BOLD)
                    .text_color(rgb(0xcdd6f4))
                    .child("CVPixelBuffer Interop Demo"),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0xa6adc8))
                    .child(self.label.clone()),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x585b70))
                    .child(self.device_info.clone()),
            )
            .child(
                div()
                    .flex()
                    .gap_4()
                    // --- Direct CVPixelBuffer path ---
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(rgb(0x9399b2))
                                    .child("Direct — surface(cv_pixel_buffer)"),
                            )
                            .child(
                                div()
                                    .w(px(320.))
                                    .h(px(240.))
                                    .bg(rgb(0x313244))
                                    .border_1()
                                    .border_color(rgb(0x45475a))
                                    .rounded_md()
                                    .overflow_hidden()
                                    .child(if let Some(ref pb) = self.test_pixel_buffer {
                                        surface(pb.clone())
                                            .object_fit(ObjectFit::Contain)
                                            .into_any_element()
                                    } else {
                                        fallback("Creating pixel buffer...").into_any_element()
                                    }),
                            ),
                    )
                    // --- wgpu import path ---
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(rgb(0x9399b2))
                                    .child("wgpu Import — surface((texture, desc))"),
                            )
                            .child(
                                div()
                                    .w(px(320.))
                                    .h(px(240.))
                                    .bg(rgb(0x313244))
                                    .border_1()
                                    .border_color(rgb(0x45475a))
                                    .rounded_md()
                                    .overflow_hidden()
                                    .child(
                                        if let (Some(tex), Some(desc)) =
                                            (&self.imported_texture, &descriptor)
                                        {
                                            surface((tex.clone(), *desc))
                                                .object_fit(ObjectFit::Contain)
                                                .into_any_element()
                                        } else {
                                            fallback("Importing via wgpu...").into_any_element()
                                        },
                                    ),
                            ),
                    ),
            )
            .child(
                div()
                    .bg(rgb(0x313244))
                    .border_1()
                    .border_color(rgb(0x45475a))
                    .rounded_md()
                    .p_3()
                    .text_sm()
                    .text_color(rgb(0xcdd6f4))
                    .child(SharedString::from(format!(
                        "Test pattern: {}×{} BGRA8 quadrants (red|green / blue|white).  \
                         Left: surface(pixel_buffer) — zero-copy Metal.  \
                         Right: import_cv_pixel_buffer_to_wgpu() — wgpu import.  \
                         Feature: wgpu-renderer (default) + iosurface-interop (zero-copy).",
                        PB_WIDTH, PB_HEIGHT,
                    ))),
            )
    }
}

fn fallback(label: &str) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .justify_center()
        .size_full()
        .bg(rgb(0x585b70).opacity(0.3))
        .text_sm()
        .text_color(rgb(0x9399b2))
        .child(SharedString::from(label))
}

fn main() {
    env_logger::init();
    let label = SharedString::from(format!(
        "Left: direct CVPixelBuffer path — surface(pixel_buffer).  \
         Right: wgpu import path — import_cv_pixel_buffer_to_wgpu().\n\
         Test pattern: {}×{} BGRA8, four quadrants (red|green / blue|white).\n\
         Build: cargo run -p gpui_macos --example cv_interop",
        PB_WIDTH, PB_HEIGHT,
    ));

    Application::with_platform(std::rc::Rc::new(MacPlatform::new(false))).run(
        move |cx: &mut App| {
            let bounds = Bounds::centered(None, size(px(820.), px(520.0)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                |_, cx| {
                    cx.new(|_| CvInteropDemo {
                        label: label.clone(),
                        device_info: SharedString::from("(no wgpu device yet)"),
                        test_pixel_buffer: None,
                        imported_texture: None,
                    })
                },
            )
            .unwrap();
            cx.activate(true);
        },
    );
}
