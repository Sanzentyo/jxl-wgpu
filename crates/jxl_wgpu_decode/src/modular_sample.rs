//! Original sample encoding, independent of Modular working-word geometry and render state.

use jxl_gpu_bitstream::SampleBitDepth;

use crate::modular_transform::GpuModularChannelLayout;

/// Validated JPEG XL sample precision, packed as total bits and exponent bits for the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularSampleEncoding(u32);

impl ModularSampleEncoding {
    pub(crate) const fn new(depth: SampleBitDepth) -> Option<Self> {
        match depth {
            SampleBitDepth::Integer { bits_per_sample } => {
                if bits_per_sample >= 1 && bits_per_sample <= 31 {
                    Some(Self(bits_per_sample))
                } else {
                    None
                }
            }
            SampleBitDepth::Float {
                bits_per_sample,
                exponent_bits_per_sample,
            } => {
                if exponent_bits_per_sample >= 2
                    && exponent_bits_per_sample <= 8
                    && bits_per_sample >= exponent_bits_per_sample + 3
                    && bits_per_sample <= exponent_bits_per_sample + 24
                {
                    Some(Self(bits_per_sample | (exponent_bits_per_sample << 8)))
                } else {
                    None
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) const fn integer(bits: u32) -> Option<Self> {
        Self::new(SampleBitDepth::Integer {
            bits_per_sample: bits,
        })
    }

    pub(crate) const fn bits(self) -> u8 {
        (self.0 & 255) as u8
    }
    pub(crate) const fn is_float(self) -> bool {
        self.0 >> 8 != 0
    }
    pub(crate) const fn packed(self) -> u32 {
        self.0
    }

    pub(crate) const fn depth(self) -> SampleBitDepth {
        if self.is_float() {
            SampleBitDepth::Float {
                bits_per_sample: self.bits() as u32,
                exponent_bits_per_sample: self.0 >> 8,
            }
        } else {
            SampleBitDepth::Integer {
                bits_per_sample: self.bits() as u32,
            }
        }
    }
}

/// An inverse-transformed view and its independently declared source sample interpretation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularOutputPlane {
    pub layout: GpuModularChannelLayout,
    pub encoding: ModularSampleEncoding,
}

impl ModularOutputPlane {
    pub(crate) const fn new(
        layout: GpuModularChannelLayout,
        encoding: ModularSampleEncoding,
    ) -> Self {
        Self { layout, encoding }
    }
}

pub(crate) fn shader(source: &str) -> String {
    source.replace(
        "/*__JXL_MODULAR_SAMPLE__*/",
        include_str!("modular_sample.wgsl"),
    )
}

#[cfg(test)]
pub(crate) fn integer_planes(planes: &[GpuModularChannelLayout]) -> Vec<ModularOutputPlane> {
    planes
        .iter()
        .map(|&layout| {
            ModularOutputPlane::new(
                layout,
                ModularSampleEncoding::integer(layout.bit_depth).unwrap(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_abi_rejects_unrepresentable_declarations() {
        for (depth, packed) in [
            (
                SampleBitDepth::Integer {
                    bits_per_sample: 16,
                },
                0x10,
            ),
            (
                SampleBitDepth::Float {
                    bits_per_sample: 16,
                    exponent_bits_per_sample: 5,
                },
                0x510,
            ),
            (
                SampleBitDepth::Float {
                    bits_per_sample: 32,
                    exponent_bits_per_sample: 8,
                },
                0x820,
            ),
        ] {
            let encoding = ModularSampleEncoding::new(depth).unwrap();
            assert_eq!(encoding.depth(), depth);
            assert_eq!(encoding.packed(), packed);
        }
        for bits_per_sample in [0, 32, u32::MAX] {
            assert!(
                ModularSampleEncoding::new(SampleBitDepth::Integer { bits_per_sample }).is_none()
            );
        }
        for (bits_per_sample, exponent_bits_per_sample) in [
            (4, 2),
            (27, 2),
            (32, 7),
            (16, 1),
            (16, 9),
            (0, 0),
            (u32::MAX, 8),
            (32, u32::MAX),
        ] {
            assert!(
                ModularSampleEncoding::new(SampleBitDepth::Float {
                    bits_per_sample,
                    exponent_bits_per_sample
                })
                .is_none()
            );
        }
    }

    #[test]
    fn unsigned_quantization_and_alpha_rescaling_are_exact_on_gpu() {
        use wgpu::util::DeviceExt;
        let Ok(backend) =
            pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default()))
        else {
            return;
        };
        let mut records = Vec::<[u32; 4]>::new();
        let mut expected = Vec::new();
        for source_bits in 1..=31 {
            let source = (1u32 << source_bits) - 1;
            for target_bits in 1..=31 {
                let target = (1u32 << target_bits) - 1;
                for sample in [0, 1, source, source - 1, source / 2, source / 2 + 1]
                    .into_iter()
                    .chain((0u32..24).map(|i| i.wrapping_mul(0x9e3779b9) & source))
                {
                    records.push([0, sample, source, target]);
                    expected.push(
                        ((u64::from(sample) * u64::from(target) + u64::from(source / 2))
                            / u64::from(source)) as u32,
                    );
                }
            }
            let mut samples = vec![
                0.0f32,
                -0.0,
                -1.0,
                f32::NAN,
                f32::INFINITY,
                1.0,
                f32::from_bits(1),
            ];
            for code in [0, 1, source / 3, source / 2, source - 1] {
                let center = ((f64::from(code) + 0.5) / f64::from(source)) as f32;
                samples.extend([center.next_down(), center, center.next_up()]);
            }
            samples.extend((90..128).flat_map(|exponent| {
                [0, 1, 0x3fffff, 0x7fffff]
                    .map(|mantissa| f32::from_bits((exponent << 23) | mantissa))
            }));
            for sample in samples {
                records.push([1, sample.to_bits(), source, 0]);
                expected.push(if sample.is_nan() || sample <= 0.0 {
                    0
                } else {
                    if sample >= 1.0 {
                        source
                    } else {
                        // Even f64 can round a 55-bit product onto the wrong side of a half-code
                        // boundary. Evaluate the exact binary32 rational with independent u128 arithmetic.
                        let word = sample.to_bits();
                        let numerator =
                            u128::from((word & 0x7fffff) | 0x800000) * u128::from(source);
                        1u128
                            .checked_shl(150 - (word >> 23))
                            .map_or(0, |denominator| {
                                ((numerator + denominator / 2) / denominator) as u32
                            })
                    }
                });
            }
        }
        let device = backend.device();
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("integer quantization and alpha reference inputs"),
            contents: bytemuck::cast_slice(&records),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("integer precision GPU results"),
            size: expected.len() as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let source = shader(
            r#"
            /*__JXL_MODULAR_SAMPLE__*/
            @group(0) @binding(0) var<storage, read> records: array<vec4<u32>>;
            @group(0) @binding(1) var<storage, read_write> result: array<u32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                if id.x >= arrayLength(&records) { return; }
                let record = records[id.x];
                if record.x == 0u { result[id.x] = modular_rescale_unsigned(record.y, record.z, record.w); }
                else { result[id.x] = modular_quantize_unsigned(bitcast<f32>(record.y), record.z); }
            }
        "#,
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("integer precision conformance"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups((expected.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output.size());
        backend.queue().submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let data = staging.slice(..).get_mapped_range().unwrap();
        let actual: &[u32] = bytemuck::cast_slice(&data);
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(actual, expected, "case {index}: {:?}", records[index]);
        }
    }
}
