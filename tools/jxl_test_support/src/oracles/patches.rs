//! Scalar f64 patch evaluation used only to freeze independently reconstructed LF previews.
//! Algebra follows libjxl v0.12.0 blending.cc and alpha.cc; no production decoder is called.
use jxl_gpu_bitstream::{ExtraChannelTypeInventory, ImageHeaderInventory};

use super::lf::Planes;

#[derive(Clone, Copy)]
struct Blend {
    mode: u32,
    alpha: usize,
    clamp: bool,
}

fn channel(
    old: f64,
    incoming: f64,
    old_alpha: f64,
    new_alpha: f64,
    info: Blend,
    associated: bool,
    own_alpha: bool,
) -> f64 {
    let clamp = |v: f64| if info.clamp { v.clamp(0.0, 1.0) } else { v };
    let (bottom, top, ba, ta) = if matches!(info.mode, 5 | 7) {
        (incoming, old, new_alpha, clamp(old_alpha))
    } else {
        (old, incoming, old_alpha, clamp(new_alpha))
    };
    match info.mode {
        0 => old,
        1 => incoming,
        2 => old + incoming,
        3 => old * clamp(incoming),
        4 | 5 => {
            let alpha = 1.0 - (1.0 - ta) * (1.0 - ba);
            if own_alpha {
                alpha
            } else if associated {
                top + bottom * (1.0 - ta)
            } else if alpha > 0.0 {
                (top * ta + bottom * ba * (1.0 - ta)) / alpha
            } else {
                0.0
            }
        }
        6 | 7 => {
            if own_alpha {
                bottom
            } else {
                bottom + top * ta
            }
        }
        _ => panic!("invalid oracle blend"),
    }
}

/// Decode the fixture's independent symbol sequence, using pre-patch alpha for every channel.
pub fn apply(
    target: &mut Planes,
    reference: &Planes,
    image: &ImageHeaderInventory,
    values: &[u32],
) {
    let mut values = values.iter().copied();
    let has_alpha = image
        .extra_channels
        .iter()
        .any(|e| matches!(e.channel_type, ExtraChannelTypeInventory::Alpha { .. }));
    let extras = image.extra_channels.len();
    for _ in 0..values.next().unwrap() {
        assert_eq!(values.next(), Some(3));
        let sx = values.next().unwrap() as usize;
        let sy = values.next().unwrap() as usize;
        let width = values.next().unwrap() as usize + 1;
        let height = values.next().unwrap() as usize + 1;
        let count = values.next().unwrap() + 1;
        let mut destination = [0i64; 2];
        assert!(sx + width <= reference.width && sy + height <= reference.height);
        for index in 0..count {
            for axis in &mut destination {
                let n = values.next().unwrap();
                if index == 0 {
                    *axis = i64::from(n);
                } else {
                    *axis += i64::from(n >> 1) ^ -i64::from(n & 1);
                }
            }
            let [x, y] = destination.map(|v| usize::try_from(v).unwrap());
            assert!(x + width <= target.width && y + height <= target.height);
            let blending: Vec<_> = (0..=extras)
                .map(|_| {
                    let mode = values.next().unwrap();
                    let alpha = if mode >= 4 && extras > 1 {
                        values.next().unwrap() as usize
                    } else {
                        0
                    };
                    let clamp = mode >= 3 && values.next().unwrap() != 0;
                    Blend { mode, alpha, clamp }
                })
                .collect();
            for row in 0..height {
                for col in 0..width {
                    let to = (y + row) * target.width + x + col;
                    let from = (sy + row) * reference.width + sx + col;
                    let old: Vec<_> = target.channels.iter().map(|p| p[to]).collect();
                    let incoming: Vec<_> = reference.channels.iter().map(|p| p[from]).collect();
                    let mut output = old.clone();
                    for c in 0..3 + extras {
                        let info = blending[if c < 3 { 0 } else { c - 2 }];
                        output[c] = if info.mode < 4 {
                            channel(old[c], incoming[c], 0.0, 0.0, info, false, false)
                        } else if c < 3 && !has_alpha {
                            if matches!(info.mode, 4 | 5) {
                                incoming[c]
                            } else {
                                old[c] + incoming[c]
                            }
                        } else {
                            channel(
                                old[c],
                                incoming[c],
                                old[3 + info.alpha],
                                incoming[3 + info.alpha],
                                info,
                                matches!(
                                    image.extra_channels[info.alpha].channel_type,
                                    ExtraChannelTypeInventory::Alpha { associated: true }
                                ),
                                c == 3 + info.alpha,
                            )
                        };
                    }
                    let color = blending[0];
                    if has_alpha && matches!(color.mode, 4 | 5) {
                        let c = 3 + color.alpha;
                        output[c] =
                            channel(old[c], incoming[c], old[c], incoming[c], color, false, true);
                    }
                    for (plane, value) in target.channels.iter_mut().zip(output) {
                        plane[to] = value;
                    }
                }
            }
        }
    }
    assert!(values.next().is_none());
}

pub fn packed(planes: Planes, image: &ImageHeaderInventory) -> Vec<f32> {
    let pixels = planes.width * planes.height;
    let planes = planes.into_linear(image);
    let alpha = image
        .extra_channels
        .iter()
        .position(|e| matches!(e.channel_type, ExtraChannelTypeInventory::Alpha { .. }));
    let mut output = Vec::with_capacity(pixels * (planes.len() + 1));
    for (i, &red) in planes[0].iter().enumerate() {
        output.extend([red as f32, planes[1][i] as f32, planes[2][i] as f32]);
        output.push(alpha.map_or(1.0, |c| planes[3 + c][i] as f32));
    }
    output.extend(planes[3..].iter().flat_map(|p| p.iter().map(|v| *v as f32)));
    output
}
