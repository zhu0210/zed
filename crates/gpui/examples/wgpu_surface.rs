#![cfg_attr(target_family = "wasm", no_main)]

//! Demonstrates the `surface()` element with a real wgpu texture.
//!
//! Creates a colorful triangle in a wgpu texture (CPU-side rendering with
//! barycentric interpolation), then composites it into the UI via
//! `surface()`.  Uses the new `Window::gpu_context()` API to obtain the
//! renderer's wgpu device.
//!
//! ```sh
//! cargo run -p gpui --example wgpu_surface
//! ```

use gpui::{
    App, Bounds, Context, DevicePixels, GpuContextHandle, GpuTextureColorSpace,
    GpuTextureDescriptor, GpuTextureFormat, IntoElement, ObjectFit, ParentElement, Render,
    SharedString, Styled, Window, WindowBounds, WindowOptions, div, prelude::*, px, rgb,
    size, surface,
};
use gpui_platform::application;
#[cfg(feature = "wgpu")]
use std::sync::Arc;

struct SurfaceDemo {
    label: SharedString,
    #[cfg(feature = "wgpu")]
    triangle_texture: Option<Arc<wgpu::Texture>>,
}

const TEX_WIDTH: u32 = 256;
const TEX_HEIGHT: u32 = 256;

/// Build the descriptor that matches our hand-drawn triangle texture.
fn make_descriptor() -> GpuTextureDescriptor {
    GpuTextureDescriptor {
        size: size(
            DevicePixels::from(TEX_WIDTH as i32),
            DevicePixels::from(TEX_HEIGHT as i32),
        ),
        format: GpuTextureFormat::Rgba8Unorm,
        color_space: GpuTextureColorSpace::Srgb,
    }
}

/// Fill an RGBA8 buffer with a colourful triangle using barycentric
/// interpolation.  Corners are red (top), green (bottom-left), blue
/// (bottom-right).
#[cfg(feature = "wgpu")]
fn fill_triangle_pixels(pixels: &mut [u8], width: u32, height: u32) {
    // Triangle fills nearly the entire canvas with minimal padding.
    let margin = 4.0;
    let v0 = (width as f32 * 0.5, margin);                     // top    → red
    let v1 = (margin, height as f32 - margin);                 // left   → green
    let v2 = (width as f32 - margin, height as f32 - margin);  // right  → blue

    // Colours at each vertex (premultiplied alpha, sRGB-ish for demo).
    let c0 = (1.0f32, 0.2, 0.2);
    let c1 = (0.2, 1.0, 0.2);
    let c2 = (0.2, 0.2, 1.0);

    let area = edge_function(v0, v1, v2);
    if area <= 0.0 {
        return; // degenerate triangle
    }

    for y in 0..height {
        for x in 0..width {
            let p = (x as f32 + 0.5, y as f32 + 0.5);

            let w0 = edge_function(v1, v2, p) / area;
            let w1 = edge_function(v2, v0, p) / area;
            let w2 = edge_function(v0, v1, p) / area;

            let idx = ((y * width + x) * 4) as usize;
            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                let r = (w0 * c0.0 + w1 * c1.0 + w2 * c2.0).clamp(0.0, 1.0);
                let g = (w0 * c0.1 + w1 * c1.1 + w2 * c2.1).clamp(0.0, 1.0);
                let b = (w0 * c0.2 + w1 * c1.2 + w2 * c2.2).clamp(0.0, 1.0);
                pixels[idx] = (r * 255.0) as u8;
                pixels[idx + 1] = (g * 255.0) as u8;
                pixels[idx + 2] = (b * 255.0) as u8;
                pixels[idx + 3] = 255;
            } else {
                // Transparent background.
                pixels[idx] = 0;
                pixels[idx + 1] = 0;
                pixels[idx + 2] = 0;
                pixels[idx + 3] = 0;
            }
        }
    }
}

/// Edge function from a to b, evaluated at point c.
#[cfg(feature = "wgpu")]
fn edge_function(
    a: (f32, f32),
    b: (f32, f32),
    c: (f32, f32),
) -> f32 {
    (c.0 - a.0) * (b.1 - a.1) - (c.1 - a.1) * (b.0 - a.0)
}

/// Build an RGBA8 wgpu texture containing a colourful triangle.
#[cfg(feature = "wgpu")]
fn create_triangle_texture(gpu: &GpuContextHandle) -> Arc<wgpu::Texture> {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("triangle_texture"),
        size: wgpu::Extent3d {
            width: TEX_WIDTH,
            height: TEX_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let mut pixels = vec![0u8; (TEX_WIDTH * TEX_HEIGHT * 4) as usize];
    fill_triangle_pixels(&mut pixels, TEX_WIDTH, TEX_HEIGHT);

    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(TEX_WIDTH * 4),
            rows_per_image: Some(TEX_HEIGHT),
        },
        wgpu::Extent3d {
            width: TEX_WIDTH,
            height: TEX_HEIGHT,
            depth_or_array_layers: 1,
        },
    );

    Arc::new(texture)
}

impl Render for SurfaceDemo {
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        #[cfg(feature = "wgpu")]
        if self.triangle_texture.is_none() {
            if let Some(gpu) = window.gpu_context() {
                self.triangle_texture = Some(create_triangle_texture(&gpu));
            }
        }

        let descriptor = make_descriptor();

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
                    .child("GPUI Surface Demo — Real wgpu Texture"),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0xa6adc8))
                    .child(self.label.clone()),
            )
            .child(
                // Row: triangle texture with different object-fit modes
                div()
                    .flex()
                    .gap_4()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(rgb(0x9399b2))
                                    .child("surface() — ObjectFit::Contain (200×120)"),
                            )
                            .child(
                                div()
                                    .w(px(200.))
                                    .h(px(120.))
                                    .bg(rgb(0x313244))
                                    .border_1()
                                    .border_color(rgb(0x45475a))
                                    .rounded_md()
                                    .overflow_hidden()
                                    .child(
                                        #[cfg(feature = "wgpu")]
                                        if let Some(ref tex) = self.triangle_texture {
                                            surface((
                                                tex.clone(),
                                                Some(descriptor.size),
                                            ))
                                            .object_fit(ObjectFit::Contain)
                                            .into_any_element()
                                        } else {
                                            placeholder_fallback("Loading...").into_any_element()
                                        },
                                        #[cfg(not(feature = "wgpu"))]
                                        placeholder_fallback("No wgpu feature").into_any_element(),
                                    ),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(rgb(0x9399b2))
                                    .child("surface() — ObjectFit::Fill (120×200)"),
                            )
                            .child(
                                div()
                                    .w(px(120.))
                                    .h(px(200.))
                                    .bg(rgb(0x313244))
                                    .border_1()
                                    .border_color(rgb(0x45475a))
                                    .rounded_md()
                                    .overflow_hidden()
                                    .child(
                                        #[cfg(feature = "wgpu")]
                                        if let Some(ref tex) = self.triangle_texture {
                                            surface(tex.clone())
                                                .object_fit(ObjectFit::Fill)
                                                .size_full()
                                                .into_any_element()
                                        } else {
                                            placeholder_fallback("Loading...").into_any_element()
                                        },
                                        #[cfg(not(feature = "wgpu"))]
                                        placeholder_fallback("No wgpu feature").into_any_element(),
                                    ),
                            ),
                    ),
            )
            .child(
                // API info
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .bg(rgb(0x313244))
                    .border_1()
                    .border_color(rgb(0x45475a))
                    .rounded_md()
                    .p_3()
                    .text_sm()
                    .text_color(rgb(0xcdd6f4))
                    .font_family("monospace")
                    .child(format!(
                        "// Texture: {}×{} RGBA8 — triangle via barycentric coords",
                        TEX_WIDTH, TEX_HEIGHT,
                    ))
                    .child("// Obtained via window.gpu_context().unwrap()")
                    .child("let texture = create_triangle_texture(&gpu);")
                    .child("")
                    .child("// Display with aspect-ratio-aware layout:")
                    .child("surface((texture.clone(), Some(descriptor.size)))")
                    .child("    .object_fit(ObjectFit::Contain)")
                    .child("")
                    .child("// Or fill the container bounds:")
                    .child("surface(texture.clone())")
                    .child("    .object_fit(ObjectFit::Fill)"),
            )
    }
}

fn placeholder_fallback(label: &str) -> impl IntoElement {
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

fn run_example() {
    let label = SharedString::from(format!(
        "Real wgpu texture ({}×{} RGBA8) composited via surface().\n\
         Triangle is CPU-rendered with barycentric interpolation, uploaded to GPU texture.\n\
         Uses Window::gpu_context() to access the renderer's wgpu device.",
        TEX_WIDTH, TEX_HEIGHT,
    ));

    application().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(720.), px(500.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|_| SurfaceDemo {
                    label: label.clone(),
                    #[cfg(feature = "wgpu")]
                    triangle_texture: None,
                })
            },
        )
        .unwrap();
        cx.activate(true);
    });
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    run_example();
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    gpui_platform::web_init();
    run_example();
}
