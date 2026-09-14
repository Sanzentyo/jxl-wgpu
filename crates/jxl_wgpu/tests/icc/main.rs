#![cfg(not(target_arch = "wasm32"))]

use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::mpsc;

use jxl_gpu_protocol::{
    Extent2d,
    icc::{IccProfile, IccRenderingIntent, IccTransform},
};
use jxl_wgpu::{
    DirectReadbackPolicy, KernelVariant, ResidentIccDispatch, ResidentIccInputs,
    ResidentIccMemoryPlan, ResidentIccPipeline, ResidentIccPlane, ResidentIccProgram,
    ResidentStorageBinding, WgpuBackend, WgpuBackendConfig,
};
use serde::Deserialize;
use wgpu::util::DeviceExt;

mod analytic;
mod black;
mod connection;
mod device_output;
mod intents;
mod linear;
mod lut;
mod metadata;
mod mpe;
mod rgb;
mod samples;

#[derive(Deserialize)]
struct Manifest {
    width: u32,
    height: u32,
    profiles: Vec<ProfileRecord>,
}

#[derive(Deserialize)]
struct ProfileRecord {
    name: String,
    channels: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Reference {
    native: f32,
    exact: f32,
    lower: f32,
    upper: f32,
    native_lower: f32,
    native_upper: f32,
    native_semantics: u32,
}

fn references(name: &str) -> Vec<Reference> {
    let bytes = std::fs::read(directory().join(name)).unwrap();
    assert!(bytes.len().is_multiple_of(28));
    bytes
        .as_chunks::<28>()
        .0
        .iter()
        .map(|record| bytemuck::pod_read_unaligned(record))
        .collect()
}

fn directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data/icc")
}

fn floats(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(directory().join(name)).unwrap();
    assert!(bytes.len().is_multiple_of(4));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect()
}

fn profile(name: &str) -> IccProfile {
    IccProfile::parse(
        std::fs::read(directory().join(format!("{name}.icc")))
            .unwrap()
            .into(),
        Default::default(),
    )
    .unwrap()
}

fn backend() -> Option<WgpuBackend> {
    match pollster::block_on(WgpuBackend::request_default(WgpuBackendConfig {
        enable_timestamps: false,
        direct_readback_policy: DirectReadbackPolicy::Disabled,
        ..Default::default()
    })) {
        Ok(backend) => Some(backend),
        Err(jxl_wgpu::Error::NoAdapter) => {
            eprintln!("skipping ICC GPU test: no compatible adapter");
            None
        }
        Err(error) => panic!("ICC GPU backend: {error}"),
    }
}

const GUARD: f32 = -12345.25;

struct Storage {
    buffer: wgpu::Buffer,
    offset: u64,
    values: Vec<f32>,
    planes: Vec<ResidentIccPlane>,
}

impl Storage {
    fn new(
        backend: &WgpuBackend,
        extent: Extent2d,
        channels: usize,
        samples: Option<&[f32]>,
        padding: u32,
    ) -> Self {
        let alignment = u64::from(
            backend
                .device()
                .limits()
                .min_storage_buffer_offset_alignment,
        )
        .max(4);
        let stride = extent.width + padding;
        let length = stride * extent.height + 7;
        let planes = (0..channels)
            .map(|i| ResidentIccPlane {
                offset: 3 + i as u32 * length,
                stride,
            })
            .collect::<Vec<_>>();
        let mut values = vec![GUARD; 11 + channels * length as usize];
        if let Some(samples) = samples {
            assert_eq!(samples.len(), extent.area().unwrap() * channels);
            for y in 0..extent.height {
                for x in 0..extent.width {
                    for (c, plane) in planes.iter().enumerate() {
                        values[(plane.offset + y * plane.stride + x) as usize] =
                            samples[((y * extent.width + x) as usize) * channels + c];
                    }
                }
            }
        }
        let mut contents = vec![GUARD; alignment as usize / 4];
        contents.extend_from_slice(&values);
        contents.extend_from_slice(&[GUARD; 9]);
        let buffer = backend
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ICC guarded test planes"),
                contents: bytemuck::cast_slice(&contents),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        Self {
            buffer,
            offset: alignment,
            values,
            planes,
        }
    }

    fn binding(&self) -> ResidentStorageBinding<'_> {
        ResidentStorageBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: NonZeroU64::new(self.values.len() as u64 * 4).unwrap(),
        }
    }
}

fn run(
    backend: &WgpuBackend,
    pipeline: &ResidentIccPipeline,
    transform: &IccTransform,
    extent: Extent2d,
    input: &[f32],
    padding: u32,
) -> Vec<f32> {
    let plan = ResidentIccMemoryPlan::new(transform, &backend.device().limits()).unwrap();
    let program = ResidentIccProgram::new(backend.device(), transform).unwrap();
    assert_eq!(program.memory_plan(), plan);
    assert_eq!(plan.dispatch_bytes, 320);
    run_program(backend, pipeline, &program, extent, input, padding)
}

fn run_program(
    backend: &WgpuBackend,
    pipeline: &ResidentIccPipeline,
    program: &ResidentIccProgram,
    extent: Extent2d,
    input: &[f32],
    padding: u32,
) -> Vec<f32> {
    run_encoded_program(
        backend,
        pipeline,
        program,
        extent,
        input,
        padding,
        [jxl_wgpu::ResidentIccSampleEncoding::Direct; 2],
    )
}

fn run_encoded_program(
    backend: &WgpuBackend,
    pipeline: &ResidentIccPipeline,
    program: &ResidentIccProgram,
    extent: Extent2d,
    input: &[f32],
    padding: u32,
    encoding: [jxl_wgpu::ResidentIccSampleEncoding; 2],
) -> Vec<f32> {
    let source = Storage::new(
        backend,
        extent,
        program.input_channels(),
        Some(input),
        padding,
    );
    let mut target = Storage::new(
        backend,
        extent,
        program.output_channels(),
        None,
        padding + 3,
    );
    let staging = backend.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("ICC oracle readback"),
        size: target.buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = backend.device().create_command_encoder(&Default::default());
    let dispatch = pipeline
        .encode(
            backend.device(),
            &mut encoder,
            program,
            ResidentIccInputs {
                input_encoding: encoding[0],
                output_encoding: encoding[1],
                input: source.binding(),
                output: target.binding(),
                extent,
                input_planes: &source.planes,
                output_planes: &target.planes,
            },
        )
        .unwrap();
    encoder.copy_buffer_to_buffer(&target.buffer, 0, &staging, 0, target.buffer.size());
    let submission = backend.queue().submit([encoder.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    let status = dispatch.validation_buffer().map(|buffer| {
        let (send, receive) = mpsc::sync_channel(1);
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = send.send(result);
        });
        (buffer, receive)
    });
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    receiver.recv().unwrap().unwrap();
    if let Some((buffer, receive)) = status {
        receive.recv().unwrap().unwrap();
        let bytes = buffer.slice(..).get_mapped_range().unwrap();
        ResidentIccDispatch::validate_status(&bytes).unwrap();
        drop(bytes);
        buffer.unmap();
    }
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    let data: &[f32] = bytemuck::cast_slice(&mapped);
    let start = target.offset as usize / 4;
    assert!(data[..start].iter().all(|v| *v == GUARD));
    assert!(
        data[start + target.values.len()..]
            .iter()
            .all(|v| *v == GUARD)
    );
    let mut result = Vec::new();
    for y in 0..extent.height {
        for x in 0..extent.width {
            for plane in &target.planes {
                let index = (plane.offset + y * plane.stride + x) as usize;
                result.push(data[start + index]);
                target.values[index] = data[start + index];
            }
        }
    }
    assert_eq!(
        &data[start..start + target.values.len()],
        target.values,
        "ICC must not modify padding or other planes"
    );
    drop(mapped);
    staging.unmap();
    drop(dispatch);
    result
}

#[test]
fn all_native_matrix_trc_profiles_match_independent_colorimetric_references() {
    let Some(backend) = backend() else {
        return;
    };
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(directory().join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest.profiles.len(), 10);
    let extent = Extent2d::new(manifest.width, manifest.height);
    let pipeline = ResidentIccPipeline::new(backend.device()).unwrap();
    let mut worst = Vec::new();
    let mut total = 0;
    let mut native_differences = [0; 4];
    for source in &manifest.profiles {
        let input = floats(&format!("{}_input.f32le", source.name));
        assert_eq!(input.len(), extent.area().unwrap() * source.channels);
        let source_profile = profile(&source.name);
        for target in &manifest.profiles {
            let transform = IccTransform::new(
                &source_profile,
                &profile(&target.name),
                IccRenderingIntent::Relative,
            )
            .unwrap();
            let actual = run(&backend, &pipeline, &transform, extent, &input, 5);
            let expected = references(&format!("{}_to_{}.reference", source.name, target.name));
            assert_eq!(actual.len(), extent.area().unwrap() * target.channels);
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, reference)) in actual.iter().zip(&expected).enumerate() {
                assert!(actual.is_finite());
                assert!(reference.lower <= reference.exact && reference.exact <= reference.upper);
                assert!(
                    *actual >= reference.lower && *actual <= reference.upper,
                    "{} -> {}, sample {index}: GPU {actual}, independent reference {reference:?}",
                    source.name,
                    target.name
                );
                assert!(reference.native_semantics < 4);
                native_differences[reference.native_semantics as usize] += 1;
                if reference.native_semantics == 0 {
                    assert!(
                        reference.native >= reference.native_lower
                            && reference.native <= reference.native_upper,
                        "{} -> {}, sample {index}: native outside independently propagated precision: {reference:?}",
                        source.name,
                        target.name
                    );
                }
                total += 1;
            }
            let (index, error) = actual
                .iter()
                .zip(&expected)
                .enumerate()
                .map(|(i, (a, e))| (i, (a - e.exact).abs()))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            worst.push((
                error,
                source.name.clone(),
                target.name.clone(),
                index,
                actual[index],
                expected[index].exact,
            ));
        }
    }
    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
    eprintln!(
        "ICC {total} components; native semantics counts {native_differences:?}; largest absolute errors (all within propagated bounds): {:#?}",
        &worst[..10]
    );
    assert_eq!(total, 176120);
    assert!(native_differences[0] > 170000);
    assert!(native_differences[1] > 0 && native_differences[2] > 0);
}
