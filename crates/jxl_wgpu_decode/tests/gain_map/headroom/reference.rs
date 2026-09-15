use super::super::*;

pub(super) struct Map {
    pub metadata: GainMapMetadata,
    pub code: Vec<u8>,
    pub pixels: Vec<f64>,
    pub extent: [usize; 2],
    tolerance: f64,
    bounds: [Vec<f64>; 2],
}

impl Map {
    pub fn load(index: usize) -> Self {
        let name = format!("case_{index}");
        let bytes = std::fs::read(directory().join(format!("{name}.jxl"))).unwrap();
        let parsed = jxl_gpu_bitstream::parse(&bytes, Default::default()).unwrap();
        let payload = parsed
            .auxiliary_boxes()
            .iter()
            .find(|b| b.box_type == JHGM)
            .unwrap()
            .payload;
        let bundle = GainMapBundle::parse(payload, Default::default()).unwrap();
        let inventory = jxl_gpu_bitstream::parse(bundle.codestream(), Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let header = &inventory.image_header;
        assert_eq!(inventory.frames.len(), 1);
        let metadata = bundle.metadata().clone();
        assert!(metadata.use_base_color_space);
        // The pre-existing original-color corpus contract, selected by actual codestream syntax.
        let tolerance = if inventory.frames[0].encoding == jxl_gpu_bitstream::FrameEncoding::Modular
            && !header.xyb_encoded
        {
            1e-5
        } else {
            1.0 / 1024.0
        };
        let pixels = floats(&format!("{name}.gain.f32"));
        let bounds = [-1.0, 1.0].map(|sign| {
            pixels
                .iter()
                .map(|v| v + sign * tolerance * (1.0 + v.abs()))
                .collect()
        });
        Self {
            metadata,
            code: bundle.codestream().to_vec(),
            pixels,
            extent: [header.width as usize, header.height as usize],
            tolerance,
            bounds,
        }
    }

    pub fn container(&self, base: &[u8]) -> Vec<u8> {
        let code = jxl_gpu_bitstream::parse(base, Default::default()).unwrap();
        let payload = GainMapBundle::new(
            self.metadata.clone(),
            &[],
            &[],
            &self.code,
            Default::default(),
        )
        .unwrap()
        .encode(Default::default())
        .unwrap();
        jxl_gpu_bitstream::write_container_with_boxes(
            code.codestream(),
            &[jxl_gpu_bitstream::ContainerBox {
                box_type: JHGM,
                payload: &payload,
            }],
        )
        .unwrap()
    }

    pub fn check_pixels(&self, actual: &[u32]) {
        assert_eq!(actual.len(), self.pixels.len());
        for (i, (&word, &expected)) in actual.iter().zip(&self.pixels).enumerate() {
            let bound = if i % 4 == 3 { 2e-6 } else { self.tolerance };
            let value = f64::from(f32::from_bits(word));
            assert!(
                value.is_finite() && (value - expected).abs() <= bound * (1.0 + expected.abs()),
                "gain sample {i}: {value} vs {expected}, normalized bound {bound}"
            );
        }
    }
}

pub(super) struct Reference<'a> {
    pub map: &'a Map,
    pub extent: [usize; 2],
    pub weight: f64,
    pub scale: f64,
}

impl Reference<'_> {
    pub fn rgb(&self, rgb: [f64; 3], position: [usize; 2]) -> [f64; 3] {
        if self.weight == 0.0 {
            return rgb;
        }
        std::array::from_fn(|c| {
            let m = self.map.metadata.channels[c];
            let gain = sample(&self.map.pixels, self.map.extent, self.extent, position, c)
                .powf(1.0 / m.gamma.value());
            let gain = (self.weight * (m.min.value() * (1.0 - gain) + m.max.value() * gain)).exp2();
            ((rgb[c] * self.scale + m.base_offset.value()) * gain - m.alternate_offset.value())
                / self.scale
        })
    }

    pub fn interval(&self, rgb: [[f64; 2]; 3], position: [usize; 2]) -> [[f64; 2]; 3] {
        if self.weight == 0.0 {
            return rgb;
        }
        std::array::from_fn(|c| {
            let m = self.map.metadata.channels[c];
            // Bilinear interpolation has nonnegative weights. Interpolate the predeclared
            // per-texel bounds, then carry them through clamp, gamma and the signed exponent.
            let gains = self.map.bounds.each_ref().map(|pixels| {
                let g = sample(pixels, self.map.extent, self.extent, position, c)
                    .powf(1.0 / m.gamma.value());
                (self.weight * (m.min.value() * (1.0 - g) + m.max.value() * g)).exp2()
            });
            // Baseline plus offset may be negative; all four products are needed.
            let products = rgb[c].map(|v| {
                gains.map(|gain| {
                    ((v * self.scale + m.base_offset.value()) * gain - m.alternate_offset.value())
                        / self.scale
                })
            });
            [
                products.into_iter().flatten().fold(f64::INFINITY, f64::min),
                products
                    .into_iter()
                    .flatten()
                    .fold(f64::NEG_INFINITY, f64::max),
            ]
        })
    }

    pub fn image(&self, base: &[f64]) -> Vec<f64> {
        base.as_chunks::<4>()
            .0
            .iter()
            .enumerate()
            .flat_map(|(pixel, rgba)| {
                let rgb = self.rgb(
                    [rgba[0], rgba[1], rgba[2]],
                    [pixel % self.extent[0], pixel / self.extent[0]],
                );
                [rgb[0], rgb[1], rgb[2], rgba[3]]
            })
            .collect()
    }

    pub fn check_native(&self, native: &native::Oracle, base: &[f64], headroom: f64) {
        let reference_units: Vec<_> = base
            .iter()
            .enumerate()
            .map(|(i, v)| if i % 4 == 3 { *v } else { v * self.scale })
            .collect();
        let actual = native.apply(
            &self.map.metadata.encode().unwrap(),
            &reference_units,
            &self.map.pixels,
            self.extent,
            self.map.extent,
            headroom,
        );
        let expected = self.image(base);
        assert_eq!(actual.len(), expected.len());
        for (i, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            let expected = if i % 4 == 3 {
                expected
            } else {
                expected * self.scale
            };
            assert!(
                (actual - expected).abs() <= 3e-6 * (1.0 + expected.abs()),
                "native weighted gain {i}: {actual} vs {expected}"
            );
        }
    }
}
