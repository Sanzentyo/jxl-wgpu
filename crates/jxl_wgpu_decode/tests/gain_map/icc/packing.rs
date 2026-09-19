use super::*;
use jxl_gpu_formats::{Channel, ImageLayout, PackingFieldKind};

#[derive(Debug)]
pub(super) struct Packing {
    pub sample: ColorSample,
    pub storage: ColorStorage,
    pub associated: bool,
    pub apply_orientation: bool,
}

impl Packing {
    pub fn request(&self, profile: IccProfile, channels: usize) -> GpuOutputRequest {
        let planar = self.storage == ColorStorage::Planar;
        let color = ColorSpecification::Icc(profile.clone());
        let mut format = if self.sample == ColorSample::F32 && channels == 1 {
            PixelFormat::gray_f32(true, planar, color)
        } else if self.sample == ColorSample::F32 && channels == 3 {
            PixelFormat::rgb_f32(RgbChannelOrder::Rgba, planar, color)
        } else {
            PixelFormat::icc_device(profile, self.sample, self.storage, true).unwrap()
        };
        // Device descriptors allow arbitrary physical component order. Named RGB/Gray use
        // their public constructor's classified layout, including its canonical X/Y/Z/W order.
        if format.model == jxl_gpu_formats::ColorModel::IccDevice {
            if planar {
                format.planes.reverse();
            } else {
                format.planes[0].words.reverse();
            }
        }
        GpuOutputRequest::color(format)
            .unwrap()
            .with_alpha_output_policy(if self.associated {
                AlphaOutputPolicy::Associated
            } else {
                AlphaOutputPolicy::Unassociated
            })
            .with_orientation_policy(if self.apply_orientation {
                OrientationPolicy::Apply
            } else {
                OrientationPolicy::Keep
            })
    }

    pub fn check(
        &self,
        case: &cases::Case,
        channels: usize,
        expected: &[[f32; 6]],
        layout: &ImageLayout,
        bytes: &[u8],
        intent: IccRenderingIntent,
    ) -> usize {
        let [width, height] = case.extent;
        let orientation = if self.apply_orientation {
            case.orientation
        } else {
            1
        };
        let extent = if orientation >= 5 {
            [height, width]
        } else {
            [width, height]
        };
        assert_eq!(
            [layout.extent.width as usize, layout.extent.height as usize],
            extent
        );
        assert_eq!(bytes.len(), layout.logical_size as usize);
        assert_eq!(case.alpha.len(), width * height);
        let mut components = 0;
        for (plane, format) in layout.planes.iter().zip(&layout.format.planes) {
            for (position, word) in format.words.iter().enumerate() {
                let PackingFieldKind::Channel(channel) = word.fields[0].kind else {
                    unreachable!()
                };
                let component = match channel {
                    Channel::Device(c) => Some(usize::from(c)),
                    Channel::X => Some(0),
                    Channel::Y => Some(1),
                    Channel::Z => Some(2),
                    Channel::W | Channel::Alpha => None,
                };
                for p in 0..width * height {
                    let (x, y) = (p % width, p / width);
                    let (ox, oy) = match orientation {
                        1 => (x, y),
                        2 => (width - 1 - x, y),
                        3 => (width - 1 - x, height - 1 - y),
                        4 => (x, height - 1 - y),
                        5 => (y, x),
                        6 => (height - 1 - y, x),
                        7 => (height - 1 - y, width - 1 - x),
                        8 => (y, width - 1 - x),
                        _ => unreachable!(),
                    };
                    let [low, high] = if let Some(c) = component {
                        assert!(c < channels);
                        let range = [expected[p * channels + c][2], expected[p * channels + c][3]]
                            .map(f64::from);
                        if self.associated {
                            let alpha = case.alpha[p].map(|v| v.max(1.0 / 67108864.0));
                            let products = range.map(|v| alpha.map(|a| a * v));
                            let low = products.into_iter().flatten().fold(f64::INFINITY, f64::min);
                            let high = products
                                .into_iter()
                                .flatten()
                                .fold(f64::NEG_INFINITY, f64::max);
                            let round = 2.0 * f64::from(f32::EPSILON) * low.abs().max(high.abs());
                            [low - round, high + round]
                        } else {
                            range
                        }
                    } else {
                        case.alpha[p]
                    };
                    let step = usize::from(word.fields[0].bits / 8);
                    let offset = plane.offset as usize
                        + oy * plane.row_stride as usize
                        + (ox * format.words.len() + position) * step;
                    let (value, low, high) = if self.sample == ColorSample::F32 {
                        (
                            f64::from(f32::from_le_bytes(
                                bytes[offset..offset + 4].try_into().unwrap(),
                            )),
                            low,
                            high,
                        )
                    } else {
                        let quantize = |v: f64| (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor();
                        (f64::from(bytes[offset]), quantize(low), quantize(high))
                    };
                    assert!(
                        value.is_finite() && low <= value && value <= high,
                        "{} {intent:?} {self:?} {x},{y},{channel:?}: GPU {value}, [{low}, {high}]",
                        case.name
                    );
                    components += 1;
                }
            }
        }
        assert_eq!(components, width * height * (channels + 1));
        components
    }
}
