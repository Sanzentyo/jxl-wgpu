use super::*;

fn parameters() -> ResidentNoiseParameters {
    ResidentNoiseParameters {
        lut: [0.01, 0.08, 0.03, 0.02, 0.04, 0.11, 0.07, 0.06],
        correlation: [-0.375, 1.25],
        frame_seed: [1, 0],
        group_dimension: 256,
    }
}

#[test]
fn portable_shader_and_admission_cover_model_and_address_limits() {
    let module = naga::front::wgsl::parse_str(include_str!("../../shaders/noise.wgsl")).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let limits = wgpu::Limits::default();
    let plan = ResidentNoisePlan::new(Extent2d::new(257, 17), parameters(), &limits).unwrap();
    assert_eq!(plan.total_bytes(), 257 * 17 * 12 + 96);
    for extent in [
        Extent2d::new(0, 1),
        Extent2d::new(1, 0),
        Extent2d::new(u32::MAX, 1),
        Extent2d::new(65536, 65536),
    ] {
        assert!(ResidentNoisePlan::new(extent, parameters(), &limits).is_err());
    }
    for value in [f32::NAN, f32::INFINITY, -0.1, 1.0] {
        let mut invalid = parameters();
        invalid.lut[3] = value;
        assert!(ResidentNoisePlan::new(Extent2d::new(1, 1), invalid, &limits).is_err());
    }
    let limited = wgpu::Limits {
        max_storage_buffer_binding_size: plan.storage_bytes - 1,
        ..limits
    };
    assert!(matches!(
        ResidentNoisePlan::new(Extent2d::new(257, 17), parameters(), &limited),
        Err(ResidentNoiseError::Limit {
            resource: "scratch bytes",
            ..
        })
    ));
}

// Scalar u64 oracle follows libjxl's eight-lane Xorshift128Plus and Random3Planes traversal.
// This intentionally uses native u64 arithmetic instead of the shader's pair multiplication.
fn reference_random(width: usize, height: usize, parameters: ResidentNoiseParameters) -> Vec<f32> {
    fn mix(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    let pixels = width * height;
    let mut random = vec![0.0; pixels * 3];
    let group = parameters.group_dimension as usize;
    for y0 in (0..height).step_by(group) {
        for x0 in (0..width).step_by(group) {
            let mut state = [[0_u64; 2]; 8];
            state[0] = [
                mix(((u64::from(parameters.frame_seed[0]) << 32)
                    | u64::from(parameters.frame_seed[1]))
                .wrapping_add(0x9e3779b97f4a7c15)),
                mix((((x0 as u64) << 32) | y0 as u64).wrapping_add(0x9e3779b97f4a7c15)),
            ];
            for i in 1..8 {
                state[i] = state[i - 1].map(mix);
            }
            for c in 0..3 {
                for y in y0..height.min(y0 + group) {
                    for x in (x0..width.min(x0 + group)).step_by(16) {
                        for (lane, [a, b]) in state.iter_mut().enumerate() {
                            let bits = a.wrapping_add(*b);
                            let mut next = *a;
                            *a = *b;
                            next ^= next << 23;
                            *b = next ^ *b ^ (next >> 18) ^ (*b >> 5);
                            for (part, word) in
                                [bits as u32, (bits >> 32) as u32].into_iter().enumerate()
                            {
                                let xx = x + lane * 2 + part;
                                if xx < width.min(x0 + group) {
                                    random[c * pixels + y * width + xx] =
                                        f32::from_bits((word >> 9) | 0x3f800000);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    random
}

fn convolved(random: &[f32], width: usize, height: usize, x: usize, y: usize, c: usize) -> f32 {
    let at = |dx: isize, dy: isize| {
        let mirror = |v: isize, n: usize| {
            let v = v.rem_euclid(n as isize * 2) as usize;
            if v < n { v } else { 2 * n - 1 - v }
        };
        random[c * width * height
            + mirror(y as isize + dy, height) * width
            + mirror(x as isize + dx, width)]
    };
    let mut others = 0.0;
    for dx in -2..=2 {
        for dy in [-2, -1, 1, 2] {
            others += at(dx, dy);
        }
    }
    for dx in [-2, -1, 1, 2] {
        others += at(dx, 0);
    }
    (others * 0.16 + at(0, 0) * -3.84) * 0.22
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_noise_matches_u64_reference_across_tiles_seeds_and_partial_row_tails() {
    let backend = match pollster::block_on(crate::WgpuBackend::request_default(
        crate::WgpuBackendConfig {
            enable_timestamps: false,
            ..Default::default()
        },
    )) {
        Ok(backend) => backend,
        Err(crate::Error::NoAdapter) => return,
        Err(error) => panic!("noise test adapter: {error}"),
    };
    let device = backend.device();
    let pipeline = ResidentNoisePipeline::new(device).unwrap();
    for (width, height, dimension, seed) in [
        (1, 1, 128, [1, 0]),
        (2, 3, 256, [1, 2]),
        (17, 5, 512, [2, 0]),
        (257, 259, 256, [1, 0]),
        (131, 129, 128, [u32::MAX, u32::MAX]),
        (1025, 2, 1024, [17, 11]),
    ] {
        let parameters = ResidentNoiseParameters {
            group_dimension: dimension,
            frame_seed: seed,
            ..parameters()
        };
        let plan =
            ResidentNoisePlan::new(Extent2d::new(width, height), parameters, &device.limits())
                .unwrap();
        let w = width as usize;
        let h = height as usize;
        let source: [Vec<f32>; 3] = std::array::from_fn(|c| {
            let stride = w + 1 + c;
            (0..stride * h)
                .map(|i| {
                    if i % stride >= w {
                        91.0
                    } else {
                        match c {
                            0 => (i % 13) as f32 * 0.1 - 0.6,
                            1 => (i % 37) as f32 * 0.1 - 0.2,
                            _ => 0.4,
                        }
                    }
                })
                .collect()
        });
        let images: [wgpu::Buffer; 3] = std::array::from_fn(|c| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("noise color reference"),
                contents: bytemuck::cast_slice(&source[c]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        });
        let scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("noise reference random"),
            size: plan.storage_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("noise aggregate reference"),
            size: plan.storage_bytes + images.iter().map(wgpu::Buffer::size).sum::<u64>(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        let _uniform = pipeline
            .encode(
                device,
                &mut encoder,
                ResidentNoiseInputs {
                    plan: &plan,
                    scratch: &scratch,
                    planes: std::array::from_fn(|c| ResidentF32Plane {
                        storage: ResidentStorageBinding::entire(&images[c]).unwrap(),
                        width,
                        height,
                        stride: width + 1 + c as u32,
                    }),
                },
            )
            .unwrap();
        encoder.copy_buffer_to_buffer(&scratch, 0, &staging, 0, plan.storage_bytes);
        let mut offset = plan.storage_bytes;
        for image in &images {
            encoder.copy_buffer_to_buffer(image, 0, &staging, offset, image.size());
            offset += image.size();
        }
        let submission = backend.queue().submit([encoder.finish()]);
        let (send, recv) = std::sync::mpsc::sync_channel(1);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = send.send(result);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .unwrap();
        recv.recv().unwrap().unwrap();
        let mapped = staging.slice(..).get_mapped_range().unwrap();
        let actual: &[f32] = bytemuck::cast_slice(&mapped);
        let random = reference_random(w, h, parameters);
        assert_eq!(
            &actual[..random.len()],
            random,
            "{width}x{height}, group={dimension}, seed={seed:?}"
        );
        let strength = |value: f32| {
            let scaled = (value * 6.0).clamp(0.0, 7.0);
            let lo = (scaled.floor() as usize).min(6);
            parameters.lut[lo]
                + (parameters.lut[lo + 1] - parameters.lut[lo]) * (scaled - lo as f32)
        };
        let mut offset = random.len();
        for c in 0..3 {
            let stride = w + 1 + c;
            for y in 0..h {
                for x in 0..stride {
                    let index = y * stride + x;
                    let expected = if x >= w {
                        source[c][index]
                    } else {
                        let vx = source[0][y * (w + 1) + x];
                        let vy = source[1][y * (w + 2) + x];
                        let rnd: [f32; 3] =
                            std::array::from_fn(|channel| convolved(&random, w, h, x, y, channel));
                        let red =
                            strength((vy + vx) * 0.5) * (rnd[0] / 128.0 + rnd[2] * (127.0 / 128.0));
                        let green =
                            strength((vy - vx) * 0.5) * (rnd[1] / 128.0 + rnd[2] * (127.0 / 128.0));
                        source[c][index]
                            + match c {
                                0 => parameters.correlation[0] * (red + green) + red - green,
                                1 => red + green,
                                _ => parameters.correlation[1] * (red + green),
                            }
                    };
                    assert!(
                        (actual[offset + index] - expected).abs() < 2e-6,
                        "{width}x{height} plane{c} ({x},{y}): actual={} expected={expected}",
                        actual[offset + index]
                    );
                }
            }
            offset += source[c].len();
        }
        drop(mapped);
        staging.unmap();
    }
}
