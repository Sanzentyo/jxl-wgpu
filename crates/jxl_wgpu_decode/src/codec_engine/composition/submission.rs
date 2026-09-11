//! Accounted GPU submission and completion lifetime shared by blending and presentation.

use crate::{Error, Result};
use jxl_wgpu::{GpuBufferLease, MemoryPermit, SubmissionPollPermit, WgpuBackend};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};
use wgpu::util::DeviceExt;

pub(super) struct Submission<'a> {
    pub pipeline: &'a wgpu::ComputePipeline,
    pub params: &'a [u8],
    pub inputs: &'a [(u32, &'a GpuBufferLease)],
    pub metadata: Option<(u32, &'a [u8])>,
    pub output_binding: u32,
    pub uniform_binding: u32,
    pub size: u64,
    pub dispatch: [u32; 2],
}

pub(super) fn submit(backend: &WgpuBackend, request: Submission<'_>) -> Result<GpuWork> {
    let Submission {
        pipeline,
        params,
        inputs,
        metadata,
        output_binding,
        uniform_binding,
        size,
        dispatch,
    } = request;
    let device = backend.device();
    if params.len() as u64 > device.limits().max_uniform_buffer_binding_size {
        return Err(Error::CompositionResourceLimit {
            resource: "uniform bytes",
            requested: params.len() as u64,
            limit: device.limits().max_uniform_buffer_binding_size,
        });
    }
    if let Some((_, bytes)) = metadata {
        validate_size(device, bytes.len() as u64)?;
    }
    let memory = backend.transient_memory_budget();
    let poll = backend.submission_poller().try_reserve()?;
    let output_permit = memory.try_reserve(size)?;
    let uniform_permit = memory.try_reserve(
        params.len() as u64
            + metadata.map_or(0, |(_, bytes)| bytes.len() as u64)
            + completion_fence_bytes(),
    )?;
    let output = GpuBufferLease::from_tracked(
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("JPEG XL composed frame"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        output_permit,
    );
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("JPEG XL composition parameters"),
        contents: params,
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut entries = inputs
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_wgpu_buffer().as_entire_binding(),
        })
        .collect::<Vec<_>>();
    entries.push(wgpu::BindGroupEntry {
        binding: output_binding,
        resource: output.as_wgpu_buffer().as_entire_binding(),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: uniform_binding,
        resource: uniform.as_entire_binding(),
    });
    let metadata_buffer = metadata.map(|(_, bytes)| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JPEG XL presentation channel metadata"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        })
    });
    if let Some((binding, _)) = metadata {
        entries.push(wgpu::BindGroupEntry {
            binding,
            resource: metadata_buffer
                .as_ref()
                .expect("metadata uploaded")
                .as_entire_binding(),
        });
    }
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("JPEG XL composition bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("JPEG XL frame composition"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("JPEG XL frame composition"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch[0], dispatch[1], 1);
    }
    submit_recorded(
        backend,
        encoder,
        output,
        inputs.iter().map(|(_, lease)| (*lease).clone()).collect(),
        (uniform, metadata_buffer),
        uniform_permit,
        poll,
    )
}

/// Completes a recorded render chain with the same accounted lifetime as frame composition.
pub(super) fn submit_recorded<R: wgpu::WasmNotSendSync + 'static>(
    backend: &WgpuBackend,
    encoder: wgpu::CommandEncoder,
    output: GpuBufferLease,
    inputs: Vec<GpuBufferLease>,
    resources: R,
    permit: MemoryPermit,
    poll: SubmissionPollPermit,
) -> Result<GpuWork> {
    let guards = inputs
        .iter()
        .map(GpuBufferLease::try_acquire_gpu_submission)
        .collect::<jxl_wgpu::Result<Vec<_>>>()?;
    let completion = Arc::new(Completion::default());
    #[cfg(target_arch = "wasm32")]
    let mut encoder = encoder;
    // wgpu work-done callbacks require Send even on WebGPU. A tiny mapped completion fence
    // uses the browser-local map callback instead, retaining the non-Send WebGPU handles.
    // Its contents are never read; no image samples cross the CPU boundary.
    #[cfg(target_arch = "wasm32")]
    let completion_fence = {
        let fence = backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("JPEG XL composition completion fence"),
            size: completion_fence_bytes(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.clear_buffer(&fence, 0, None);
        fence
    };
    let lifetime = Arc::new(WorkLifetime {
        _buffers: inputs.iter().cloned().chain([output.clone()]).collect(),
        _resources: resources,
        _permit: permit,
        #[cfg(target_arch = "wasm32")]
        completion_fence,
    });
    let done = Arc::clone(&completion);
    let retained = Arc::clone(&lifetime);
    #[cfg(not(target_arch = "wasm32"))]
    encoder.on_submitted_work_done(move || {
        drop(retained);
        done.complete(Ok(()));
    });
    let submission = backend.queue().submit([encoder.finish()]);
    drop(guards);
    #[cfg(target_arch = "wasm32")]
    lifetime
        .completion_fence
        .map_async(wgpu::MapMode::Read, .., move |result| {
            if result.is_ok() {
                retained.completion_fence.unmap();
            }
            drop(retained);
            done.complete(result.map_err(|error| error.to_string()));
        });
    let failed = Arc::clone(&completion);
    poll.register(submission, move |error| {
        #[cfg(not(target_arch = "wasm32"))]
        drop(lifetime);
        failed.complete(Err(error));
    })?;
    Ok(GpuWork {
        output: Some(output),
        completion,
    })
}

pub(super) fn validate_size(device: &wgpu::Device, size: u64) -> Result<()> {
    let limits = device.limits();
    let limit = u64::from(u32::MAX - 3)
        .min(limits.max_buffer_size)
        .min(limits.max_storage_buffer_binding_size);
    if size == 0 {
        return Err(Error::EngineContract(
            "composition requires a nonempty surface",
        ));
    }
    if size > limit {
        return Err(Error::CompositionResourceLimit {
            resource: "buffer bytes",
            requested: size,
            limit,
        });
    }
    Ok(())
}

struct WorkLifetime<R> {
    _buffers: Vec<GpuBufferLease>,
    _resources: R,
    _permit: MemoryPermit,
    #[cfg(target_arch = "wasm32")]
    completion_fence: wgpu::Buffer,
}

pub(super) const fn completion_fence_bytes() -> u64 {
    if cfg!(target_arch = "wasm32") { 4 } else { 0 }
}

#[derive(Debug, Default)]
pub(super) struct Completion {
    state: Mutex<CompletionState>,
    condition: Condvar,
}
#[derive(Debug, Default)]
struct CompletionState {
    result: Option<std::result::Result<(), String>>,
    waker: Option<Waker>,
}

impl Completion {
    pub(super) fn complete(&self, result: std::result::Result<(), String>) {
        let waker = {
            let mut state = super::lock(&self.state);
            if state.result.is_some() {
                return;
            }
            state.result = Some(result);
            state.waker.take()
        };
        self.condition.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }
    pub(super) fn poll(&self, context: &Context<'_>) -> Poll<Result<()>> {
        let mut state = super::lock(&self.state);
        if let Some(result) = state.result.as_ref() {
            return Poll::Ready(result.clone().map_err(Error::backend));
        }
        state.waker = Some(context.waker().clone());
        Poll::Pending
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn wait(&self) -> Result<()> {
        let mut state = super::lock(&self.state);
        while state.result.is_none() {
            state = self
                .condition
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state
            .result
            .as_ref()
            .expect("completion signalled")
            .clone()
            .map_err(Error::backend)
    }
}

#[derive(Debug)]
pub(super) struct GpuWork {
    output: Option<GpuBufferLease>,
    completion: Arc<Completion>,
}
impl GpuWork {
    pub(super) fn poll(&mut self, context: &Context<'_>) -> Poll<Result<GpuBufferLease>> {
        self.completion.poll(context).map(|result| {
            result?;
            self.output.take().ok_or(Error::EngineContract(
                "composition completion consumed twice",
            ))
        })
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn wait(mut self) -> Result<GpuBufferLease> {
        self.completion.wait()?;
        self.output.take().ok_or(Error::EngineContract(
            "composition completion consumed twice",
        ))
    }
    pub(super) fn unvalidated(&self) -> Result<GpuBufferLease> {
        self.output.clone().ok_or(Error::EngineContract(
            "composition completion consumed twice",
        ))
    }
}
