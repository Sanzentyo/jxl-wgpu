//! Bounded GPU feature entropy with count/validate replay and exact resident output admission.
//! The 16-byte control result contains only status, continuation and allocation counts.
use std::sync::Arc;
use std::task::{Context, Poll};

use bytemuck::Pod;
use jxl_wgpu::{GpuBufferLease, MemoryPermit, WgpuBackend};
use wgpu::util::DeviceExt;

use super::submission::{Completion, validate_size};
use crate::entropy_window::{EntropyStreamWindows, GroupStreamSegment};
use crate::{Error, GpuCodestream, Result};

pub(super) const STATE_WORDS: u64 = 32;
pub(super) const STATUS_BYTES: u64 = 16;

pub(super) trait Program: Copy + std::fmt::Debug + Pod {
    const LABEL: &'static str;
    const SHADER: &'static str;
    fn update(&mut self, window: &GroupStreamSegment, capacity: u32, reset: bool);
    fn stride(&self) -> u32;
    fn rejected(&self, code: u32) -> Error;
}

#[derive(Clone, Debug)]
pub(super) struct Plan<P: Program> {
    pub(super) metadata: Vec<u32>,
    pub(super) windows: EntropyStreamWindows,
    pub(super) token_start: u64,
    pub(super) history_words: u32,
    pub(super) params: P,
}

fn overflow() -> Error {
    Error::backend("feature entropy continuation or addressing exceeds u32")
}

impl<P: Program> Plan<P> {
    pub(super) fn submit(
        self,
        backend: WgpuBackend,
        source: Arc<GpuCodestream>,
    ) -> Result<Pending<P>> {
        let device = backend.device();
        let scratch_bytes = (STATE_WORDS + u64::from(self.history_words)) * 4;
        let stream_bytes = self.windows.stream_bytes();
        let metadata_bytes = self.metadata.len() as u64 * 4;
        for size in [scratch_bytes, stream_bytes, metadata_bytes] {
            validate_size(device, size)?;
        }
        let permit = backend.transient_memory_budget().try_reserve(
            scratch_bytes
                + stream_bytes
                + metadata_bytes
                + std::mem::size_of::<P>() as u64
                + STATUS_BYTES,
        )?;
        let command_permit = backend.transient_memory_budget().try_reserve(4)?;
        let poll = backend.submission_poller().try_reserve()?;
        let shader = P::SHADER
            .replace(
                "/*__JXL_MODULAR_ENTROPY_ABI__*/",
                include_str!("../../modular_entropy_abi.wgsl"),
            )
            .replace(
                "/*__JXL_MODULAR_ENTROPY__*/",
                include_str!("../../modular_entropy.wgsl"),
            );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(P::LABEL),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(P::LABEL),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let storage = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let commands = GpuBufferLease::from_tracked(
            storage(
                "JPEG XL feature count placeholder",
                4,
                wgpu::BufferUsages::STORAGE,
            ),
            command_permit,
        );
        let resources = Arc::new(Resources {
            stream: storage(
                "JPEG XL feature bit window",
                stream_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            metadata: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("JPEG XL feature entropy tables"),
                contents: bytemuck::cast_slice(&self.metadata),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            state: storage(
                "JPEG XL feature continuation",
                scratch_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            params: storage(
                "JPEG XL feature entropy parameters",
                std::mem::size_of::<P>() as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            status: storage(
                "JPEG XL feature control readback",
                STATUS_BYTES,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            _permit: permit,
        });
        let mut pending = Pending {
            backend,
            source,
            plan: self,
            pipeline,
            resources,
            commands,
            window: 0,
            cursor: 0,
            counting: true,
            count: 0,
            expected_end: None,
            completion: Arc::new(Completion::default()),
            submissions: 0,
        };
        pending.submit_window(poll, true)?;
        Ok(pending)
    }
}

#[derive(Debug)]
struct Resources {
    stream: wgpu::Buffer,
    metadata: wgpu::Buffer,
    state: wgpu::Buffer,
    params: wgpu::Buffer,
    status: wgpu::Buffer,
    _permit: MemoryPermit,
}

#[derive(Debug)]
pub(super) struct DecodedProgram {
    pub(super) commands: GpuBufferLease,
    pub(super) count: u32,
    pub(super) stride: u32,
    pub(super) end: u64,
}

#[derive(Debug)]
pub(super) struct Pending<P: Program> {
    backend: WgpuBackend,
    source: Arc<GpuCodestream>,
    plan: Plan<P>,
    pipeline: wgpu::ComputePipeline,
    resources: Arc<Resources>,
    commands: GpuBufferLease,
    window: usize,
    cursor: u32,
    counting: bool,
    count: u32,
    expected_end: Option<u32>,
    completion: Arc<Completion>,
    pub(super) submissions: usize,
}

impl<P: Program> Pending<P> {
    fn submit_window(&mut self, poll: jxl_wgpu::SubmissionPollPermit, reset: bool) -> Result<()> {
        let segment = self.plan.windows.get(self.window).ok_or_else(overflow)?;
        let mut bytes = vec![0u8; (segment.input_end - segment.input_start).div_ceil(4) * 4 + 4];
        self.source.copy_range(
            segment.input_start as u64..segment.input_end as u64,
            &mut bytes[..segment.input_end - segment.input_start],
        )?;
        self.backend
            .queue()
            .write_buffer(&self.resources.stream, 0, &bytes);
        let mut params = self.plan.params;
        params.update(&segment, if self.counting { 0 } else { self.count }, reset);
        self.backend
            .queue()
            .write_buffer(&self.resources.params, 0, bytemuck::bytes_of(&params));
        let buffers = [
            &self.resources.stream,
            &self.resources.metadata,
            &self.resources.state,
            self.commands.as_wgpu_buffer(),
            &self.resources.params,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bindings = self
            .backend
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(P::LABEL),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
        let mut encoder =
            self.backend
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(P::LABEL),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.resources.state,
            0,
            &self.resources.status,
            0,
            STATUS_BYTES,
        );
        let completion = Arc::new(Completion::default());
        let done = Arc::clone(&completion);
        let resources = Arc::clone(&self.resources);
        let commands = self.commands.clone();
        let submission = self.backend.queue().submit([encoder.finish()]);
        self.resources
            .status
            .map_async(wgpu::MapMode::Read, .., move |result| {
                drop((resources, commands));
                done.complete(result.map_err(|error| error.to_string()));
            });
        let failed = Arc::clone(&completion);
        poll.register(submission, move |error| failed.complete(Err(error)))?;
        self.completion = completion;
        self.submissions += 1;
        Ok(())
    }

    fn advance(&mut self) -> Result<Option<DecodedProgram>> {
        let view = self
            .resources
            .status
            .get_mapped_range(..)
            .map_err(Error::backend)?;
        let status: [u32; 4] = bytemuck::pod_read_unaligned(&view);
        drop(view);
        self.resources.status.unmap();
        if status[0] != 0 {
            return Err(self.plan.params.rejected(status[0]));
        }
        self.cursor = status[1];
        if status[2] == u32::MAX {
            if !self.counting || status[3] == 0 {
                if self.expected_end.is_some_and(|end| end != self.cursor)
                    || (!self.counting && status[3] != self.count)
                {
                    return Err(Error::EngineContract(
                        "feature replay changed its control result",
                    ));
                }
                return Ok(Some(DecodedProgram {
                    commands: self.commands.clone(),
                    count: status[3],
                    stride: self.plan.params.stride(),
                    end: self.plan.token_start + u64::from(self.cursor),
                }));
            }
            self.count = status[3];
            self.expected_end = Some(self.cursor);
            let size = u64::from(self.count) * u64::from(self.plan.params.stride()) * 4;
            validate_size(self.backend.device(), size)?;
            let permit = self.backend.transient_memory_budget().try_reserve(size)?;
            self.commands = GpuBufferLease::from_tracked(
                self.backend
                    .device()
                    .create_buffer(&wgpu::BufferDescriptor {
                        label: Some("JPEG XL resident frame feature program"),
                        size,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }),
                permit,
            );
            self.counting = false;
            self.window = 0;
            self.cursor = 0;
            self.submit_window(self.backend.submission_poller().try_reserve()?, true)?;
        } else {
            let segment = self.plan.windows.get(self.window).ok_or_else(overflow)?;
            if self.cursor >= segment.window_yield_end
                && segment.available_token_end < segment.stream_token_end
            {
                self.window += 1;
            }
            self.submit_window(self.backend.submission_poller().try_reserve()?, false)?;
        }
        Ok(None)
    }

    pub(super) fn poll(&mut self, context: &Context<'_>) -> Poll<Result<DecodedProgram>> {
        loop {
            match self.completion.poll(context) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(result) => result?,
            }
            if let Some(dictionary) = self.advance()? {
                return Poll::Ready(Ok(dictionary));
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn complete_submission(&mut self) -> Result<Option<DecodedProgram>> {
        self.completion.wait()?;
        self.advance()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn wait(mut self) -> Result<(DecodedProgram, usize)> {
        loop {
            if let Some(dictionary) = self.complete_submission()? {
                return Ok((dictionary, self.submissions));
            }
        }
    }
}
