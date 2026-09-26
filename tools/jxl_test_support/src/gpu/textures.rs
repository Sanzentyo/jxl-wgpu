//! Raw texture carriers for input-layout conformance, with poisoned adjacent subresources.
use std::sync::Arc;

use jxl_gpu_formats::ImageLayout;

/// Upload each image plane to mip 1, array layer 1 of a separate texture.
/// Carrier formats only specify byte widths; these uploads never interpret sample values.
pub fn upload_planes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &ImageLayout,
    bytes: &[u8],
) -> Vec<Arc<wgpu::Texture>> {
    layout
        .planes
        .iter()
        .zip(&layout.format.planes)
        .map(|(plane, packing)| {
            let format = match packing.bits_per_element() {
                8 => wgpu::TextureFormat::R8Uint,
                16 => wgpu::TextureFormat::R16Uint,
                32 => wgpu::TextureFormat::R32Uint,
                64 => wgpu::TextureFormat::Rg32Uint,
                128 => wgpu::TextureFormat::Rgba32Uint,
                bits => panic!("no portable texture carrier for {bits} bits"),
            };
            let width = plane
                .sample_extent
                .width
                .div_ceil(u32::from(packing.pixels_per_element));
            let height = plane.sample_extent.height;
            let texel = format.block_copy_size(None).unwrap();
            assert_eq!(plane.row_bytes, u64::from(width * texel));
            let raw: Vec<_> = (0..height)
                .flat_map(|row| {
                    let start = (plane.offset + u64::from(row) * plane.row_stride) as usize;
                    bytes[start..start + plane.row_bytes as usize]
                        .iter()
                        .copied()
                })
                .collect();
            let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("independent texture plane with poisoned mip/layer neighbors"),
                size: wgpu::Extent3d {
                    width: width * 2,
                    height: height * 2,
                    depth_or_array_layers: 3,
                },
                mip_level_count: 2,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }));
            for mip in 0..2 {
                let width = width * (2 >> mip);
                let height = height * (2 >> mip);
                for layer in 0..3 {
                    let poison = vec![0x53 + layer as u8; (width * height * texel) as usize];
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &texture,
                            mip_level: mip,
                            origin: wgpu::Origin3d {
                                x: 0,
                                y: 0,
                                z: layer,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        if mip == 1 && layer == 1 {
                            &raw
                        } else {
                            &poison
                        },
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(width * texel),
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
            texture
        })
        .collect()
}
