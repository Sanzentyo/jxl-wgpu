//! GPU-decoded patch commands. Only status, allocation counts and the continuation bit cursor
//! cross the host boundary; positions, blend operations and pixels stay resident.

use std::sync::Arc;
use std::task::{Context, Poll};

use bytemuck::{Pod, Zeroable};
use jxl_gpu_bitstream::{FrameInventory, FrameSectionKind};
use jxl_wgpu::{GpuBufferLease, MemoryPermit, WgpuBackend};
use wgpu::util::DeviceExt;

use super::submission::{Completion, validate_size};
use crate::entropy::EntropyStreamParams;
use crate::entropy_window::{EntropyStreamWindows, GroupEntropyRange};
use crate::modular_tree::{EntropyDecoderIr, MaTreeLimits};
use crate::{Error, GpuCodestream, Result};

mod render;
pub(super) use render::render;
pub(super) use render::render_lf;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

const STATE_WORDS: u64 = 32;
const STATUS_BYTES: u64 = 16;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Params {
    entropy: EntropyStreamParams,
    capacity: u32,
    window: [u32; 4],
    image: [u32; 4],
    limits: [u32; 4],
    references: [[u32; 4]; 4],
}
const _: () = assert!(std::mem::size_of::<Params>() == 128);

#[derive(Clone, Debug)]
pub(super) struct Plan {
    metadata: Vec<u32>,
    windows: EntropyStreamWindows,
    token_start: u64,
    history_words: u32,
    params: Params,
}

impl Plan {
    pub(super) fn new(
        source: &GpuCodestream,
        frame: &FrameInventory,
        extras: &[jxl_gpu_bitstream::ExtraChannelInventory],
        references: [[u32; 4]; 4],
        limit: u64,
    ) -> Result<Self> {
        let extra_count = u32::try_from(extras.len()).map_err(|_| overflow())?;
        let section = frame
            .sections
            .iter()
            .find(|section| {
                matches!(
                    section.kind,
                    FrameSectionKind::Single | FrameSectionKind::LowFrequencyGlobal
                )
            })
            .ok_or(Error::EngineContract("patch dictionary lacks LF-global"))?;
        let mut reader = source.reader();
        reader.skip_bits(section.bits.offset)?;
        let entropy = EntropyDecoderIr::parse(&mut reader, 10, MaTreeLimits::default())?;
        let token_start = reader.bit_offset();
        let token_end = section.bits.end().ok_or_else(overflow)?;
        let (mut width, mut height) = frame.color_sample_extent().ok_or_else(overflow)?;
        if frame.encoding == jxl_gpu_bitstream::FrameEncoding::VarDct {
            // Patches address the coded, padded image before frame upsampling. Subsampled
            // YCbCr is rejected by composition admission until its feature graph is connected.
            width = width.div_ceil(8).checked_mul(8).ok_or_else(overflow)?;
            height = height.div_ceil(8).checked_mul(8).ok_or_else(overflow)?;
        }
        let max_ref = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| 1024u64.checked_add(pixels / 4))
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(overflow)?;
        let max_positions = max_ref.checked_mul(4).ok_or_else(overflow)?;
        let stride = extra_count
            .checked_add(1)
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_add(8))
            .ok_or_else(overflow)?;
        let symbols = max_positions
            .checked_mul(stride)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(overflow)?;
        let history_words = entropy.lz77_window_words(0, symbols)?;
        let mut metadata = entropy.pack_gpu_metadata()?.words;
        let contexts = u32::try_from(metadata.len()).map_err(|_| overflow())?;
        metadata.extend(
            entropy.context_to_cluster[..10]
                .iter()
                .map(|&cluster| u32::from(cluster)),
        );
        metadata.extend(extras.iter().map(|extra| {
            u32::from(matches!(
                extra.channel_type,
                jxl_gpu_bitstream::ExtraChannelTypeInventory::Alpha { associated: true }
            ))
        }));
        let windows = EntropyStreamWindows::new(
            source.logical_bytes(),
            GroupEntropyRange {
                token_bit_offset: token_start,
                token_bit_end: token_end,
            },
            limit,
        )?;
        let first = windows.get(0).ok_or_else(overflow)?;
        Ok(Self {
            metadata,
            windows,
            token_start,
            history_words,
            params: Params {
                entropy: EntropyStreamParams {
                    token_start: 0,
                    token_end: first.stream_token_end,
                    lz77_window_mask: history_words.saturating_sub(1),
                },
                capacity: 0,
                window: [0; 4],
                image: [width, height, extra_count, max_ref],
                limits: [max_positions, stride, contexts, 1],
                references,
            },
        })
    }

    pub(super) fn submit(
        self,
        backend: WgpuBackend,
        source: Arc<GpuCodestream>,
    ) -> Result<Pending> {
        let device = backend.device();
        let scratch_bytes = (STATE_WORDS + u64::from(self.history_words)) * 4;
        let stream_bytes = self.windows.stream_bytes();
        let metadata_bytes = self.metadata.len() as u64 * 4;
        for size in [scratch_bytes, stream_bytes, metadata_bytes] {
            validate_size(device, size)?;
        }
        let permit = backend
            .transient_memory_budget()
            .try_reserve(scratch_bytes + stream_bytes + metadata_bytes + 128 + STATUS_BYTES)?;
        let command_permit = backend.transient_memory_budget().try_reserve(4)?;
        let poll = backend.submission_poller().try_reserve()?;
        let shader = include_str!("patches/decode.wgsl")
            .replace(
                "/*__JXL_MODULAR_ENTROPY_ABI__*/",
                include_str!("../../modular_entropy_abi.wgsl"),
            )
            .replace(
                "/*__JXL_MODULAR_ENTROPY__*/",
                include_str!("../../modular_entropy.wgsl"),
            );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("JPEG XL patch entropy"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("JPEG XL patch entropy"),
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
                "JPEG XL patch count placeholder",
                4,
                wgpu::BufferUsages::STORAGE,
            ),
            command_permit,
        );
        let resources = Arc::new(Resources {
            stream: storage(
                "JPEG XL patch bit window",
                stream_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            metadata: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("JPEG XL patch entropy tables"),
                contents: bytemuck::cast_slice(&self.metadata),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            state: storage(
                "JPEG XL patch continuation",
                scratch_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            params: storage(
                "JPEG XL patch entropy parameters",
                128,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            status: storage(
                "JPEG XL patch control readback",
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

fn overflow() -> Error {
    Error::backend("patch dictionary geometry or addressing exceeds u32")
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
pub(super) struct Dictionary {
    pub(super) commands: GpuBufferLease,
    pub(super) count: u32,
    pub(super) stride: u32,
    pub(super) end: u64,
}

#[derive(Debug)]
pub(super) struct Pending {
    backend: WgpuBackend,
    source: Arc<GpuCodestream>,
    plan: Plan,
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

impl Pending {
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
        params.capacity = if self.counting { 0 } else { self.count };
        params.limits[3] = u32::from(reset);
        params.window = [
            segment.window_logical_start,
            segment.window_upload_start,
            segment.available_token_end,
            segment.window_yield_end,
        ];
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
                label: Some("JPEG XL patch entropy bindings"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
        let mut encoder =
            self.backend
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("JPEG XL patch dictionary continuation"),
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

    fn advance(&mut self) -> Result<Option<Dictionary>> {
        let view = self
            .resources
            .status
            .get_mapped_range(..)
            .map_err(Error::backend)?;
        let status: [u32; 4] = bytemuck::pod_read_unaligned(&view);
        drop(view);
        self.resources.status.unmap();
        if status[0] != 0 {
            return Err(Error::PatchDictionary { code: status[0] });
        }
        self.cursor = status[1];
        if status[2] == u32::MAX {
            if !self.counting || status[3] == 0 {
                if self.expected_end.is_some_and(|end| end != self.cursor)
                    || (!self.counting && status[3] != self.count)
                {
                    return Err(Error::EngineContract(
                        "patch replay changed its control result",
                    ));
                }
                return Ok(Some(Dictionary {
                    commands: self.commands.clone(),
                    count: status[3],
                    stride: self.plan.params.limits[1],
                    end: self.plan.token_start + u64::from(self.cursor),
                }));
            }
            self.count = status[3];
            self.expected_end = Some(self.cursor);
            let size = u64::from(self.count) * u64::from(self.plan.params.limits[1]) * 4;
            validate_size(self.backend.device(), size)?;
            let permit = self.backend.transient_memory_budget().try_reserve(size)?;
            self.commands = GpuBufferLease::from_tracked(
                self.backend
                    .device()
                    .create_buffer(&wgpu::BufferDescriptor {
                        label: Some("JPEG XL resident patch commands"),
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

    pub(super) fn poll(&mut self, context: &Context<'_>) -> Poll<Result<Dictionary>> {
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
    pub(super) fn wait(mut self) -> Result<(Dictionary, usize)> {
        loop {
            self.completion.wait()?;
            if let Some(dictionary) = self.advance()? {
                return Ok((dictionary, self.submissions));
            }
        }
    }
}

#[cfg(test)]
mod shader_tests {
    #[test]
    fn patch_shaders_validate() {
        let decode = include_str!("patches/decode.wgsl")
            .replace(
                "/*__JXL_MODULAR_ENTROPY_ABI__*/",
                include_str!("../../modular_entropy_abi.wgsl"),
            )
            .replace(
                "/*__JXL_MODULAR_ENTROPY__*/",
                include_str!("../../modular_entropy.wgsl"),
            );
        for source in [decode.as_str(), include_str!("patches/render.wgsl")] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
        }
    }
}
