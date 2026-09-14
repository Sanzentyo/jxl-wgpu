use super::*;
use jxl_gpu_protocol::{
    Extent2d,
    icc::{IccProfile, IccRenderingIntent, IccTransform},
};
use jxl_wgpu::{
    ResidentIccInputs, ResidentIccPipeline, ResidentIccPlane, ResidentIccProgram,
    ResidentStorageBinding,
};

fn recorded(backend: &WgpuBackend, status: u32) -> GpuWork {
    let device = backend.device();
    let memory = backend.transient_memory_budget();
    let directory =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../jxl_wgpu/test-data/icc");
    let profile = |name: &str| {
        IccProfile::parse(
            std::fs::read(directory.join(format!("{name}.icc")))
                .unwrap()
                .into(),
            Default::default(),
        )
        .unwrap()
    };
    let transform = IccTransform::new(
        &profile("black/lut8_xyz_3"),
        &profile("mpe/identity"),
        IccRenderingIntent::Perceptual,
    )
    .unwrap();
    let program = ResidentIccProgram::new(device, &transform).unwrap();
    let plan = program.memory_plan();
    assert_eq!(plan.validation_bytes, 4);
    let pipeline = ResidentIccPipeline::new(device).unwrap();
    let lease = |label| {
        GpuBufferLease::from_tracked(
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 12,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            memory.try_reserve(12).unwrap(),
        )
    };
    let input = lease("ICC completion test input");
    let output = lease("ICC completion test output");
    fn binding(lease: &GpuBufferLease) -> ResidentStorageBinding<'_> {
        ResidentStorageBinding {
            buffer: lease.as_wgpu_buffer(),
            offset: 0,
            size: std::num::NonZeroU64::new(12).unwrap(),
        }
    }
    let planes = [0, 1, 2].map(|offset| ResidentIccPlane { offset, stride: 1 });
    let mut encoder = device.create_command_encoder(&Default::default());
    let dispatch = pipeline
        .encode(
            device,
            &mut encoder,
            &program,
            ResidentIccInputs {
                input: binding(&input),
                output: binding(&output),
                extent: Extent2d::new(1, 1),
                input_planes: &planes,
                output_planes: &planes,
            },
        )
        .unwrap();
    // Inject the GPU status after normal recording to isolate completion/error plumbing.
    // The resident shader tests separately exercise a real singular connection.
    let injected = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ICC completion test status"),
        contents: &status.to_le_bytes(),
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    encoder.copy_buffer_to_buffer(&injected, 0, dispatch.validation_buffer().unwrap(), 0, 4);
    submit_icc_recorded(
        backend,
        encoder,
        output,
        vec![input],
        IccWork {
            resources: (program, pipeline, injected),
            dispatch: Some(dispatch),
        },
        memory
            .try_reserve(plan.program_bytes + plan.transient_bytes() + 4)
            .unwrap(),
        backend.submission_poller().try_reserve().unwrap(),
    )
    .unwrap()
}

#[test]
fn icc_validation_preserves_typed_errors_and_releases_completed_and_cancelled_work() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let memory = backend.transient_memory_budget();
    for status in [0, 1] {
        for asynchronous in [false, true] {
            let mut work = recorded(&backend, status);
            let result = if asynchronous {
                pollster::block_on(std::future::poll_fn(|context| work.poll(context)))
            } else {
                work.wait()
            };
            if status == 0 {
                drop(result.unwrap());
            } else {
                assert!(matches!(
                    result,
                    Err(Error::ResidentIcc(ResidentIccError::Precision))
                ));
            }
            assert_eq!(memory.snapshot().reserved_bytes, 0);
        }
        let work = recorded(&backend, status);
        let unvalidated = work.unvalidated().unwrap();
        drop(work);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while memory.snapshot().reserved_bytes != 12 && std::time::Instant::now() < deadline {
            backend.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(memory.snapshot().reserved_bytes, 12);
        drop(unvalidated);
        assert_eq!(memory.snapshot().reserved_bytes, 0);
    }
}
