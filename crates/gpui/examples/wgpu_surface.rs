#![cfg_attr(target_family = "wasm", no_main)]

//! Demonstrates the `surface()` element with real wgpu textures:
//!
//! 1. RGBA8 triangle (barycentric interpolation) — `surface((tex, desc))`
//! 2. NV12 (Y+CbCr planes) — `surface((y_tex, cb_cr_tex, size))`
//!
//! Uses `Window::gpu_context()` to obtain the renderer's wgpu device.
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
    device_info: SharedString,
    #[cfg(feature = "wgpu")]
    triangle_texture: Option<Arc<wgpu::Texture>>,
    #[cfg(feature = "wgpu")]
    nv12_y_texture: Option<Arc<wgpu::Texture>>,
    #[cfg(feature = "wgpu")]
    nv12_cb_cr_texture: Option<Arc<wgpu::Texture>>,
}

const TEX_WIDTH: u32 = 256;
const TEX_HEIGHT: u32 = 256;
const NV12_TEX_WIDTH: u32 = 256;
const NV12_TEX_HEIGHT: u32 = 256;

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

/// Build the descriptor for the NV12 test texture.
fn make_nv12_descriptor() -> GpuTextureDescriptor {
    GpuTextureDescriptor {
        size: size(
            DevicePixels::from(NV12_TEX_WIDTH as i32),
            DevicePixels::from(NV12_TEX_HEIGHT as i32),
        ),
        format: GpuTextureFormat::Nv12,
        color_space: GpuTextureColorSpace::Srgb,
    }
}

/// Fill NV12 Y and CbCr planes with a colour-bar test pattern.
///
/// Y plane: horizontal gradient from black (16) to white (235) — video range.
/// Cb plane: horizontal gradient from blue (16) to yellow (240).
/// Cr plane: vertical gradient from green (16) to red (240).
/// Centre is neutral grey.
#[cfg(feature = "wgpu")]
fn fill_nv12_test_pixels(
    y_plane: &mut [u8],
    cb_cr_plane: &mut [u8],
    width: u32,
    height: u32,
) {
    for row in 0..height {
        for col in 0..width {
            let y_idx = (row * width + col) as usize;
            let cb_cr_idx = y_idx * 2;

            // Y: horizontal ramp 16→235
            let y_val = 16.0 + (col as f32 / (width - 1) as f32) * (235.0 - 16.0);
            y_plane[y_idx] = y_val as u8;

            // Cb: horizontal ramp 16→240 (blue → yellow)
            let cb_val = 16.0 + (col as f32 / (width - 1) as f32) * (240.0 - 16.0);

            // Cr: vertical ramp 16→240 (green → red)
            let cr_val = 16.0 + (row as f32 / (height - 1) as f32) * (240.0 - 16.0);

            cb_cr_plane[cb_cr_idx] = cb_val as u8;
            cb_cr_plane[cb_cr_idx + 1] = cr_val as u8;
        }
    }
}

/// Create NV12 test textures: Y plane (R8Unorm) + CbCr plane (Rg8Unorm).
#[cfg(feature = "wgpu")]
fn create_nv12_test_textures(
    gpu: &GpuContextHandle,
) -> (Arc<wgpu::Texture>, Arc<wgpu::Texture>) {
    let y_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("nv12_y"),
        size: wgpu::Extent3d {
            width: NV12_TEX_WIDTH,
            height: NV12_TEX_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let cb_cr_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("nv12_cb_cr"),
        size: wgpu::Extent3d {
            width: NV12_TEX_WIDTH,
            height: NV12_TEX_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let y_len = (NV12_TEX_WIDTH * NV12_TEX_HEIGHT) as usize;
    let cb_cr_len = y_len * 2;
    let mut y_plane = vec![0u8; y_len];
    let mut cb_cr_plane = vec![0u8; cb_cr_len];
    fill_nv12_test_pixels(&mut y_plane, &mut cb_cr_plane, NV12_TEX_WIDTH, NV12_TEX_HEIGHT);

    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &y_texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        &y_plane,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(NV12_TEX_WIDTH),
            rows_per_image: Some(NV12_TEX_HEIGHT),
        },
        wgpu::Extent3d {
            width: NV12_TEX_WIDTH,
            height: NV12_TEX_HEIGHT,
            depth_or_array_layers: 1,
        },
    );

    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &cb_cr_texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        &cb_cr_plane,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(NV12_TEX_WIDTH * 2),
            rows_per_image: Some(NV12_TEX_HEIGHT),
        },
        wgpu::Extent3d {
            width: NV12_TEX_WIDTH,
            height: NV12_TEX_HEIGHT,
            depth_or_array_layers: 1,
        },
    );

    (Arc::new(y_texture), Arc::new(cb_cr_texture))
}

impl Render for SurfaceDemo {
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        #[cfg(feature = "wgpu")]
        if self.triangle_texture.is_none() || self.nv12_y_texture.is_none() {
            if let Some(gpu) = window.gpu_context() {
                // Log device info once at startup.
                if self.triangle_texture.is_none() {
                    log::info!(
                        "wgpu_surface demo: GPU device ready. \
                         format={:?}, dual_source_blending={}",
                        gpu.color_texture_format,
                        gpu.supports_dual_source_blending
                    );
                    self.device_info = SharedString::from(format!(
                        "wgpu device: color_format={:?}, dual_src_blend={}",
                        gpu.color_texture_format,
                        gpu.supports_dual_source_blending
                    ));
                    self.triangle_texture = Some(create_triangle_texture(&gpu));
                }
                if self.nv12_y_texture.is_none() {
                    log::info!(
                        "wgpu_surface demo: creating NV12 test textures ({}×{})",
                        NV12_TEX_WIDTH,
                        NV12_TEX_HEIGHT
                    );
                    let (y_tex, cb_cr_tex) = create_nv12_test_textures(&gpu);
                    self.nv12_y_texture = Some(y_tex);
                    self.nv12_cb_cr_texture = Some(cb_cr_tex);
                }
            }
        }

        let descriptor = make_descriptor();
        let nv12_descriptor = make_nv12_descriptor();

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
                    .child("GPUI Surface Demo — wgpu + NV12"),
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
                    .font_family("monospace")
                    .child(self.device_info.clone()),
            )
            .child(
                // Row 1: RGBA8 triangle — Contain + Fill
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
                                    .child("RGBA8 — ObjectFit::Contain (200×120)"),
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
                                    .child("RGBA8 — ObjectFit::Fill (120×200)"),
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
                // Row 2: NV12 YUV texture — Contain + Fill
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
                                    .child("NV12 — ObjectFit::Contain (180×180)"),
                            )
                            .child(
                                div()
                                    .w(px(180.))
                                    .h(px(180.))
                                    .bg(rgb(0x313244))
                                    .border_1()
                                    .border_color(rgb(0x45475a))
                                    .rounded_md()
                                    .overflow_hidden()
                                    .child(
                                        #[cfg(feature = "wgpu")]
                                        if let (Some(y_tex), Some(cb_cr_tex)) =
                                            (&self.nv12_y_texture, &self.nv12_cb_cr_texture)
                                        {
                                            surface((
                                                y_tex.clone(),
                                                cb_cr_tex.clone(),
                                                nv12_descriptor.size,
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
                                    .child("NV12 — ObjectFit::Fill (120×160)"),
                            )
                            .child(
                                div()
                                    .w(px(120.))
                                    .h(px(160.))
                                    .bg(rgb(0x313244))
                                    .border_1()
                                    .border_color(rgb(0x45475a))
                                    .rounded_md()
                                    .overflow_hidden()
                                    .child(
                                        #[cfg(feature = "wgpu")]
                                        if let (Some(y_tex), Some(cb_cr_tex)) =
                                            (&self.nv12_y_texture, &self.nv12_cb_cr_texture)
                                        {
                                            surface((
                                                y_tex.clone(),
                                                cb_cr_tex.clone(),
                                                nv12_descriptor.size,
                                            ))
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
                    .child("// --- RGBA8 texture (cross-platform) ---")
                    .child(format!(
                        "surface((texture.clone(), Some(descriptor.size)))  // {}×{}",
                        TEX_WIDTH, TEX_HEIGHT,
                    ))
                    .child("    .object_fit(ObjectFit::Contain)")
                    .child("")
                    .child("// --- NV12 texture (Y + CbCr planes, YCbCr→RGB in WGSL) ---")
                    .child(format!(
                        "surface((y_tex.clone(), cb_cr_tex.clone(), size))  // {}×{}",
                        NV12_TEX_WIDTH, NV12_TEX_HEIGHT,
                    ))
                    .child("    .object_fit(ObjectFit::Contain)")
                    .child("")
                    .child("// --- feature flags ---")
                    .child("// wgpu-renderer  : wgpu backend (default)")
                    .child("// iosurface-interop : zero-copy CVPixelBuffer→wgpu (macOS)"),
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
        "Row 1: RGBA8 triangle ({}×{}, barycentric).  \
         Row 2: NV12 Y+CbCr ({}×{}, colour bars).\n\
         All textures uploaded via queue.write_texture, composited via surface().\n\
         Uses Window::gpu_context() to access the renderer's wgpu device.\n\
         Build with: cargo run -p gpui --example wgpu_surface",
        TEX_WIDTH, TEX_HEIGHT, NV12_TEX_WIDTH, NV12_TEX_HEIGHT,
    ));

    application().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(720.), px(700.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|_| SurfaceDemo {
                    label: label.clone(),
                    device_info: SharedString::from("(no wgpu device yet)"),
                    #[cfg(feature = "wgpu")]
                    triangle_texture: None,
                    #[cfg(feature = "wgpu")]
                    nv12_y_texture: None,
                    #[cfg(feature = "wgpu")]
                    nv12_cb_cr_texture: None,
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
