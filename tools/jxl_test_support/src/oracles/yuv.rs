//! Independent f64 YCbCr reconstruction from logical code planes, not GPU packing or plans.
use jxl_gpu_formats::{ChromaLocation, ChromaSubsampling, ColorRange, ColorSpec, YcbcrEncoding};
use jxl_gpu_protocol::Extent2d;

pub struct CodePlanes {
    pub extent: Extent2d,
    pub subsampling: ChromaSubsampling,
    pub bits: u8,
    pub y: Vec<u16>,
    pub cb: Vec<u16>,
    pub cr: Vec<u16>,
}

impl CodePlanes {
    pub fn rgb(&self, color: ColorSpec, linear: bool) -> Vec<[f64; 3]> {
        let (dx, dy) = self.subsampling.chroma_divisors().unwrap();
        let cw = self.extent.width.div_ceil(u32::from(dx));
        let ch = self.extent.height.div_ceil(u32::from(dy));
        assert_eq!(self.y.len(), self.extent.area().unwrap());
        assert_eq!(self.cb.len(), (cw * ch) as usize);
        assert_eq!(self.cr.len(), self.cb.len());
        let max = ((1u32 << self.bits) - 1) as f64;
        let scale = (1u32 << (self.bits - 8)) as f64;
        let norm_y = |v: f64| match color.range {
            ColorRange::Full => v / max,
            ColorRange::Limited => (v - 16.0 * scale) / (219.0 * scale),
        };
        let norm_c = |v: f64| match color.range {
            ColorRange::Full => (v - (1u32 << (self.bits - 1)) as f64) / max,
            ColorRange::Limited => (v - 128.0 * scale) / (224.0 * scale),
        };
        let (kr, kb) = match color.encoding {
            YcbcrEncoding::Bt601 => (0.299, 0.114),
            YcbcrEncoding::Bt709 => (0.2126, 0.0722),
            YcbcrEncoding::Bt2020 | YcbcrEncoding::Bt2020ConstantLuminance => (0.2627, 0.0593),
            _ => panic!("unsupported oracle matrix"),
        };
        let kg = 1.0 - kr - kb;
        let position = |p: u32, d: u8, loc: ChromaLocation| {
            let offset = if d == 1 {
                0.0
            } else {
                match loc {
                    ChromaLocation::Even => 0.0,
                    ChromaLocation::Center => f64::from(d - 1) / 2.0,
                    ChromaLocation::Odd => f64::from(d - 1),
                    ChromaLocation::Both => panic!("ambiguous chroma phase"),
                }
            };
            (f64::from(p) - offset) / f64::from(d)
        };
        (0..self.extent.height)
            .flat_map(|y| (0..self.extent.width).map(move |x| (x, y)))
            .map(|(x, y)| {
                let cx = position(x, dx, color.chroma_location.horizontal);
                let cy = position(y, dy, color.chroma_location.vertical);
                let sample = |values: &[u16]| {
                    let mut result = 0.0;
                    for oy in 0..=1 {
                        for ox in 0..=1 {
                            let px = (cx.floor() as i64 + ox).clamp(0, i64::from(cw) - 1);
                            let py = (cy.floor() as i64 + oy).clamp(0, i64::from(ch) - 1);
                            let wx = if ox == 0 {
                                1.0 - (cx - cx.floor())
                            } else {
                                cx - cx.floor()
                            };
                            let wy = if oy == 0 {
                                1.0 - (cy - cy.floor())
                            } else {
                                cy - cy.floor()
                            };
                            result +=
                                wx * wy * f64::from(values[(py * i64::from(cw) + px) as usize]);
                        }
                    }
                    norm_c(result)
                };
                let yy = norm_y(f64::from(self.y[(y * self.extent.width + x) as usize]));
                let cb = sample(&self.cb);
                let cr = sample(&self.cr);
                if color.encoding == YcbcrEncoding::Bt2020ConstantLuminance {
                    assert!(linear);
                    let r = super::color::to_linear(
                        yy + cr * if cr <= 0.0 { 1.7184 } else { 0.9936 },
                        color.transfer,
                    );
                    let b = super::color::to_linear(
                        yy + cb * if cb <= 0.0 { 1.9404 } else { 1.5816 },
                        color.transfer,
                    );
                    let l = super::color::to_linear(yy, color.transfer);
                    [r, (l - kr * r - kb * b) / kg, b]
                } else {
                    let rgb = [
                        yy + 2.0 * (1.0 - kr) * cr,
                        yy - (2.0 * kb * (1.0 - kb) / kg) * cb - (2.0 * kr * (1.0 - kr) / kg) * cr,
                        yy + 2.0 * (1.0 - kb) * cb,
                    ];
                    if linear {
                        rgb.map(|v| super::color::to_linear(v, color.transfer))
                    } else {
                        rgb
                    }
                }
            })
            .collect()
    }
}
