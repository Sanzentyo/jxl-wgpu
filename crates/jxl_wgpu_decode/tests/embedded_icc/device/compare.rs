use super::*;
use jxl_gpu_formats::{Channel, PackingFieldKind};

pub(super) struct Oracle<'a> {
    pub case: &'a Case,
    pub original: &'a [f32],
    pub expected: &'a [[f32; 6]],
    pub passthrough: bool,
    pub intent: IccRenderingIntent,
    pub spots: bool,
}

impl Oracle<'_> {
    fn bounds(&self, pixel: usize, channel: Channel, packing: Packing) -> (f32, f32) {
        let case = self.case;
        let alpha = case.alpha(self.original, pixel);
        let Channel::Device(c) = channel else {
            assert_eq!(channel, Channel::Alpha);
            return (alpha, alpha);
        };
        let c = usize::from(c);
        let (lower, upper) = if self.passthrough {
            let index = if case.channels == 4 && c == 3 {
                3 + case.black
            } else {
                c
            };
            let v = self.original[pixel * case.source_stride() + index];
            let value = if case.channels == 4 { 1.0 - v } else { v };
            let error = if case.mode != 0 && (case.channels != 4 || c < 3) {
                2e-5
            } else {
                0.0
            };
            (value - error, value + error)
        } else {
            let bounds = self.expected[pixel * case.target_channels + c];
            (bounds[2], bounds[3])
        };
        if packing.associated {
            let multiplier = alpha.max(1.0 / 67108864.0);
            // One final binary32 multiplication, independently of the ICC bounds.
            let error = 2.0 * f32::EPSILON * lower.abs().max(upper.abs()) * multiplier;
            (lower * multiplier - error, upper * multiplier + error)
        } else {
            (lower, upper)
        }
    }

    pub fn assert_frame(
        &self,
        layout: &ImageLayout,
        packing: Packing,
        frame: usize,
        bytes: &[u8],
    ) -> usize {
        assert_eq!(bytes.len(), layout.logical_size as usize);
        let width = layout.extent.width as usize;
        let count = layout.extent.area().unwrap();
        let mut components = 0;
        for (plane, format) in layout.planes.iter().zip(&layout.format.planes) {
            for (position, word) in format.words.iter().enumerate() {
                let PackingFieldKind::Channel(channel) = word.fields[0].kind else {
                    unreachable!()
                };
                for p in 0..count {
                    let (lower, upper) = self.bounds(frame * count + p, channel, packing);
                    let step = usize::from(word.fields[0].bits / 8);
                    let offset = plane.offset as usize
                        + p / width * plane.row_stride as usize
                        + ((p % width) * format.words.len() + position) * step;
                    let (actual, low, high) = if packing.sample == ColorSample::F32 {
                        (
                            f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()),
                            lower,
                            upper,
                        )
                    } else {
                        let quantize = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor();
                        (f32::from(bytes[offset]), quantize(lower), quantize(upper))
                    };
                    assert!(
                        actual.is_finite() && low <= actual && actual <= high,
                        "{} {:?} spots {} passthrough {} {:?} {:?} frame {frame} pixel {p} {channel:?}: GPU {actual}, [{low}, {high}]",
                        self.case.name,
                        self.intent,
                        self.spots,
                        self.passthrough,
                        packing.sample,
                        packing.storage
                    );
                    components += usize::from(matches!(channel, Channel::Device(_)));
                }
            }
        }
        components
    }
}
