use std::sync::Arc;

use jxl_gpu_bitstream::{InventoryLimits, ParseLimits};
use jxl_gpu_formats::{ImageLayout, PitchLinearPlaneLayout};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu_decode::vardct::packet::{
    BoundedVarDctPacketPlan, GpuVarDctPacketStatus, VarDctModularParams, VarDctPacketBuffers,
    VarDctPacketPipeline,
};
use jxl_wgpu_encode::{
    BufferImageSource, VarDctColorEncoding, VarDctEncoder, VarDctStrategy, WgpuContext,
};
use wgpu::util::DeviceExt;

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("jxl-wgpu VarDCT packet oracle"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()
}

fn black_source(context: &WgpuContext) -> BufferImageSource {
    let extent = Extent2d::new(8, 8);
    let bytes = vec![0u8; 8 * 8 * 3];
    let layout = ImageLayout::from_planes(
        extent,
        VarDctColorEncoding::SrgbD65.pixel_format(),
        vec![PitchLinearPlaneLayout {
            plane_index: 0,
            offset: 0,
            row_stride: 24,
            sample_extent: extent,
            row_bytes: 24,
        }],
    )
    .unwrap();
    let buffer = Arc::new(
        context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("black VarDCT packet oracle source"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE,
            }),
    );
    BufferImageSource::new(buffer, layout).unwrap()
}

#[test]
fn gpu_decodes_fixed_standard_packet_entropy_and_validates_zero_ac() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping VarDCT packet GPU oracle: no adapter");
        return;
    };
    let context = WgpuContext::new(Arc::new(device.clone()), Arc::new(queue.clone())).unwrap();
    let codestream = VarDctEncoder::new(context.clone(), VarDctStrategy::Dct8)
        .unwrap()
        .encode(black_source(&context))
        .unwrap();
    let parsed = jxl_gpu_bitstream::parse(&codestream, ParseLimits::default()).unwrap();
    let inventory = parsed
        .codestream_inventory(InventoryLimits {
            max_frames: 1,
            max_total_section_bytes: codestream.len() as u64,
            ..InventoryLimits::default()
        })
        .unwrap();
    let plan = BoundedVarDctPacketPlan::parse(&codestream, &inventory).unwrap();
    let control = plan.packet_control().unwrap();

    let mut stream_bytes = codestream.clone();
    stream_bytes.resize((stream_bytes.len() + 7) & !3, 0);
    let stream = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("VarDCT packet codestream"),
        contents: &stream_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("VarDCT packet MA metadata"),
        contents: bytemuck::cast_slice(&plan.modular_metadata),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let lf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VarDCT packet LF samples"),
        size: u64::from(plan.reconstructed_words().unwrap()) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let raw = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VarDCT packet raw HF metadata"),
        size: u64::from(control.capacities[1]) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let coefficients = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VarDCT packet zero AC coefficients"),
        size: u64::from(plan.coefficient_words()) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let status = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VarDCT packet status"),
        size: std::mem::size_of::<GpuVarDctPacketStatus>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let control_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("VarDCT packet control"),
        contents: bytemuck::bytes_of(&control),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("VarDCT packet Modular params"),
        contents: bytemuck::bytes_of(
            &VarDctModularParams::default().with_lz77_window(plan.lz77_window_words),
        ),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VarDCT packet status readback"),
        size: std::mem::size_of::<GpuVarDctPacketStatus>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let pipeline = VarDctPacketPipeline::new(&device);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("VarDCT packet oracle"),
    });
    pipeline.encode(
        &device,
        &mut encoder,
        VarDctPacketBuffers {
            codestream: &stream,
            modular_metadata: &metadata,
            reconstructed_lf: &lf,
            raw_hf_metadata: &raw,
            coefficients: &coefficients,
            status: &status,
            control: &control_buffer,
            modular_params: &params,
        },
    );
    encoder.copy_buffer_to_buffer(
        &status,
        0,
        &staging,
        0,
        std::mem::size_of::<GpuVarDctPacketStatus>() as u64,
    );
    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    let status = bytemuck::from_bytes::<GpuVarDctPacketStatus>(&mapped).to_owned();
    drop(mapped);
    staging.unmap();
    let block_count = control.geometry[2] * control.geometry[3];
    status
        .validate(
            plan.transform,
            block_count * 3,
            block_count + 4,
            plan.global_scale,
            plan.quant_lf,
        )
        .unwrap();
    assert_eq!(status.coefficient_words, plan.coefficient_words());
}
