//! One image-wide routing plan for all scalar sources, including packed legacy alpha.
use super::*;
use crate::source::{SourceChannels, SourceLayout, SourceWindows};
use crate::{BufferImageSource, sample_format::ImageSamplePlan};
use jxl_gpu_protocol::Extent2d;

#[derive(Clone, Copy)]
pub(in crate::vardct_encoder) struct Limits {
    pub(in crate::vardct_encoder) buffer: u64,
    pub(in crate::vardct_encoder) binding: u64,
    pub(in crate::vardct_encoder) workgroups: u32,
    pub(in crate::vardct_encoder) alignment: u64,
}

#[derive(Clone, Copy, Debug)]
struct BoundPlane {
    plan: Plan,
    windows: SourceWindows,
    // None means the packed alpha in the main color source.
    source: Option<usize>,
    offset: u64,
}

#[derive(Clone, Debug)]
pub(in crate::vardct_encoder) struct ImagePlan {
    planes: Vec<BoundPlane>,
    pub(in crate::vardct_encoder) memory: VarDctExtraChannelMemoryPlan,
    pub(in crate::vardct_encoder) packed_alpha_memory: Option<VarDctAlphaMemoryPlan>,
    pub(in crate::vardct_encoder) source_bytes: u64,
}

pub(in crate::vardct_encoder) fn pass_for_shift(progression: &ProgressivePlan, shift: u8) -> u32 {
    progression
        .downsampling()
        .iter()
        .find(|point| u32::from(point.factor) <= 1u32 << shift)
        .map_or(progression.passes().len() as u32 - 1, |point| {
            u32::from(point.last_pass)
        })
}

impl ImagePlan {
    pub(in crate::vardct_encoder) fn new(
        frame: super::super::types::VarDctFrameLayout,
        samples: &ImageSamplePlan,
        source: &BufferImageSource,
        main: &SourceLayout,
        progressive: &ProgressivePlan,
        code: &VarDctPrefixCode,
        limits: Limits,
    ) -> Result<Self, EncodeError> {
        let packed = usize::from(samples.alpha.is_some());
        if source.extra_channels().len() + packed != samples.extra_channels.len() {
            return Err(EncodeError::InvalidSource(
                "extra source count differs from the image declaration",
            ));
        }
        let mut global_prefix = true;
        let mut planes = Vec::with_capacity(samples.extra_channels.len());
        let mut memory = VarDctExtraChannelMemoryPlan {
            parameter_bytes: 0,
            artifact_bytes: 0,
            readback_bytes: 0,
            total_bytes: 0,
        };
        let extent = Extent2d::new(frame.width, frame.height);
        for (index, definition) in samples.extra_channels.iter().enumerate() {
            let input = index.checked_sub(packed);
            let independent;
            let (layout, component) = if let Some(input) = input {
                let scalar = &source.extra_channels()[input];
                if scalar.layout.extent != definition.source_extent(extent)
                    || !scalar.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                {
                    return Err(EncodeError::InvalidSource(
                        "extra source extent or buffer usage differs from its declaration",
                    ));
                }
                independent =
                    SourceLayout::new(&scalar.layout, scalar.buffer.size(), limits.alignment)?;
                let precision = definition.precision().color(crate::ColorChannels::Gray);
                if independent.spec.format != SourceChannels::Gray
                    || independent.spec.bits_per_sample != precision.bits_per_sample()
                    || independent.spec.exponent_bits_per_sample != precision.exponent_bits()
                {
                    return Err(EncodeError::Unsupported(UnsupportedFeature::InputFormat));
                }
                (&independent, 0)
            } else {
                (main, samples.alpha_component().expect("packed alpha"))
            };
            let plane_extent = definition.source_extent(extent);
            let windows = layout.full_windows;
            windows.validate(limits.binding)?;
            let region = layout.region(0, 0, plane_extent.width, plane_extent.height)?;
            let mut components = [region.components[component]];
            windows.rebase(&mut components, [region.offsets[component], 0, 0, 0])?;
            // Global Modular stops at the first oversized channel. Later small channels
            // still belong to their LF/pass streams; the serializer must not rediscover this.
            global_prefix &= plane_extent.width <= GROUP_DIM && plane_extent.height <= GROUP_DIM;
            let shift = definition.dimension_shift();
            let (route, group_dim) = if global_prefix {
                (Route::Global, GROUP_DIM)
            } else if shift >= 3 {
                (Route::Lf, 2048 >> shift)
            } else {
                (
                    Route::Pass(pass_for_shift(progressive, shift)),
                    GROUP_DIM >> shift,
                )
            };
            let plan = Plan::for_channel(
                Geometry {
                    extent: plane_extent,
                    group_dim,
                    route,
                    channel_index: index as u32,
                },
                components[0],
                definition.precision().mask(),
                layout.spec.big_endian,
                code,
            )?;
            plan.validate_limits(limits.buffer, limits.binding, limits.workgroups)?;
            planes.push(BoundPlane {
                plan,
                windows,
                source: input,
                offset: memory.artifact_bytes,
            });
            memory.parameter_bytes += plan.memory.parameter_bytes;
            memory.artifact_bytes += plan.memory.artifact_bytes;
            memory.readback_bytes += plan.memory.readback_bytes;
            memory.total_bytes += plan.memory.total_bytes;
        }
        let source_bytes = SourceWindows::addressed_bytes_many(
            std::iter::once((source.buffer.as_ref(), main.full_windows)).chain(
                planes.iter().filter_map(|plane| {
                    plane.source.map(|index| {
                        (
                            source.extra_channels()[index].buffer.as_ref(),
                            plane.windows,
                        )
                    })
                }),
            ),
        )?;
        let packed_alpha_memory = (packed != 0).then(|| planes[0].plan.memory);
        Ok(Self {
            planes,
            memory,
            packed_alpha_memory,
            source_bytes,
        })
    }

    pub(in crate::vardct_encoder) fn encode(
        &self,
        pipeline: &Pipeline,
        device: &wgpu::Device,
        commands: &mut wgpu::CommandEncoder,
        source: &BufferImageSource,
        readback: &wgpu::Buffer,
        offset: u64,
    ) -> Vec<Scratch> {
        self.planes
            .iter()
            .map(|plane| {
                let buffer = plane.source.map_or(&source.buffer, |index| {
                    &source.extra_channels()[index].buffer
                });
                pipeline.encode(
                    device,
                    commands,
                    plane.plan,
                    plane
                        .windows
                        .entries(buffer, super::super::dispatch::SOURCE_BINDINGS),
                    readback,
                    offset + plane.offset,
                )
            })
            .collect()
    }

    pub(in crate::vardct_encoder) fn validate<'a>(
        &'a self,
        bytes: &'a [u8],
        code: &VarDctPrefixCode,
    ) -> Result<Fragments<'a>, BackendError> {
        if bytes.len() as u64 != self.memory.artifact_bytes {
            return Err(BackendError::InvalidArtifact(
                "extra-channel artifact extent differs from plan",
            ));
        }
        for plane in &self.planes {
            plane.plan.validate(
                &bytes[plane.offset as usize
                    ..(plane.offset + plane.plan.memory.artifact_bytes) as usize],
                code,
            )?;
        }
        Ok(Fragments {
            bytes,
            plan: Some(self),
        })
    }
}

#[derive(Clone, Copy, Default)]
pub(in crate::vardct_encoder) struct Fragments<'a> {
    bytes: &'a [u8],
    plan: Option<&'a ImagePlan>,
}

impl Fragments<'_> {
    pub(in crate::vardct_encoder) fn len(self) -> usize {
        self.plan.map_or(0, |plan| plan.planes.len())
    }

    pub(in crate::vardct_encoder) fn write_global(
        self,
        output: &mut BitWriter,
    ) -> Result<(), EncodeError> {
        if self.plan.is_some() {
            self.write_stream(output, Route::Global, 0, true)?;
        }
        Ok(())
    }

    pub(in crate::vardct_encoder) fn write_lf(
        self,
        output: &mut BitWriter,
        group: u32,
    ) -> Result<(), EncodeError> {
        self.write_stream(output, Route::Lf, group, false)
    }

    pub(in crate::vardct_encoder) fn write_group(
        self,
        output: &mut BitWriter,
        group: u32,
        pass: u32,
    ) -> Result<(), EncodeError> {
        self.write_stream(output, Route::Pass(pass), group, false)
    }

    fn write_stream(
        self,
        output: &mut BitWriter,
        route: Route,
        group: u32,
        global: bool,
    ) -> Result<(), EncodeError> {
        let Some(plan) = self.plan else {
            return Ok(());
        };
        let mut selected = plan
            .planes
            .iter()
            .filter(|plane| plane.plan.route == route)
            .peekable();
        if global || selected.peek().is_some() {
            super::super::bitstream::write_local_modular_header(output)?;
        }
        for plane in selected {
            let bytes = &self.bytes
                [plane.offset as usize..(plane.offset + plane.plan.memory.artifact_bytes) as usize];
            let words = bytemuck::try_cast_slice(bytes).map_err(|_| {
                BackendError::Invariant("validated extra artifact alignment changed")
            })?;
            PlaneFragments {
                words,
                plan: Some(plane.plan),
            }
            .append_group(output, group)?;
        }
        Ok(())
    }
}
