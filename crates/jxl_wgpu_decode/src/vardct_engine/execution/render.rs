//! Reconstruct and present one immutable coefficient state. Entropy accumulation is separate.

use super::*;

pub(super) struct FrameRenderInputs<'a> {
    pub(super) source: &'a VarDctSource,
    pub(super) group_buffers: &'a [VarDctGroupJobBuffers],
    pub(super) resources: &'a wgpu::Buffer,
    pub(super) output: &'a wgpu::Buffer,
    pub(super) resident_planes: Option<&'a [wgpu::Buffer; 3]>,
    pub(super) rendered_extra: Option<&'a crate::modular_render::ModularRenderBuffers>,
    pub(super) post_transform: PostTransformJobBuffers,
    pub(super) transient_permit: &'a mut MemoryPermit,
}

pub(super) struct FrameRenderResult {
    pub(super) output_scratch: FrameOutputScratch,
    pub(super) post_transform_buffers: PostTransformJobBuffers,
    pub(super) lf_planes: Option<ProgressiveDcXybPlanes>,
    pub(super) resident_scratch: Vec<ResidentVarDctScratch>,
}

pub(super) fn encode_frame_render(
    device: &wgpu::Device,
    commands: &mut wgpu::CommandEncoder,
    pipelines: &VarDctPipelines,
    inputs: FrameRenderInputs<'_>,
) -> Result<FrameRenderResult, VarDctDecodeError> {
    let FrameRenderInputs {
        source,
        group_buffers,
        resources,
        output,
        resident_planes,
        rendered_extra,
        post_transform,
        transient_permit,
    } = inputs;
    let PostTransformJobBuffers {
        _pre_restoration_planes: pre_restoration_planes,
        _restoration_planes: restoration_planes,
        _frame_upsample_planes: frame_upsample_planes,
        _frame_upsample_weights: frame_upsample_weights,
        _epf_sigma: epf_sigma,
        _epf_sigma_uniforms: epf_sigma_uniforms,
        ..
    } = post_transform;
    let [blocks_x, blocks_y] = source.packet.block_extent();
    let padded_width = blocks_x
        .checked_mul(8)
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "padded output width",
        })?;
    let mut resident_scratch = Vec::with_capacity(source.groups.len());
    let (output_scratch, post_transform_buffers, lf_planes) = match source.output {
        VarDctFrameOutput::Color { mut config, plan } => {
            let resident_planes =
                resident_planes.ok_or(VarDctDecodeError::EntropyWindowContract {
                    detail: "color output lacks its resident planes",
                })?;
            let padded_height =
                blocks_y
                    .checked_mul(8)
                    .ok_or(VarDctDecodeError::ArithmeticOverflow {
                        field: "padded output height",
                    })?;
            let correlation_width = source.packet.profile.width.div_ceil(64);
            let correlation_height = source.packet.profile.height.div_ceil(64);

            for ((packet_group, group), buffers) in source
                .packet
                .groups
                .iter()
                .zip(&source.groups)
                .zip(group_buffers)
            {
                resident_scratch.push(pipelines.renderer.encode(
                    device,
                    commands,
                    ResidentVarDctInputs {
                        coefficients: resident_binding(&buffers.coefficients)?,
                        artifact: resident_binding(&buffers.artifact)?,
                        resources: resident_binding(resources)?,
                        outputs: resident_shifted_image_planes(
                            resident_planes,
                            padded_width,
                            padded_height,
                            source.packet.profile.channel_shifts,
                        )?,
                        indirect: &buffers.artifact,
                        indirect_base_offset: u64::from(
                            group.artifact_layout.indirect_offset_words,
                        ) * 4,
                        config: ResidentVarDctRenderConfig {
                            task_capacity: packet_group.task_capacity,
                            scratch_scalars: packet_group.coefficient_words(),
                            task_word_offset: group.artifact_layout.tasks_offset_words,
                            bucket_word_offset: group.artifact_layout.buckets_offset_words,
                            quant_offset: group.quant_offset,
                            correlation_offset: source.resource_layout.correlation_offset,
                            lf_offsets: source.resource_layout.lf_offsets,
                            lf_strides: source.resource_layout.lf_strides,
                            correlation_width,
                            correlation_height,
                            quant_biases: source.quant_biases,
                        },
                    },
                )?);
            }
            let image_width = source.packet.profile.width;
            let image_height = source.packet.profile.height;
            let mut pre_restoration_uniforms = Vec::new();
            if let Some(upsampled) = &pre_restoration_planes {
                for channel in 0..3 {
                    let shift = source.packet.profile.channel_shifts[channel];
                    if !shift.is_subsampled() {
                        continue;
                    }
                    let [input_width, input_height] = shift
                        .shifted_extent(image_width, image_height)
                        .ok_or(VarDctDecodeError::ArithmeticOverflow {
                            field: "pre-restoration input extent",
                        })?;
                    pre_restoration_uniforms.push(pipelines.chroma_upsample.encode(
                        device,
                        commands,
                        ResidentChromaUpsampleInputs {
                            input: ResidentF32Plane {
                                storage: resident_binding(&resident_planes[channel])?,
                                width: input_width,
                                height: input_height,
                                stride: padded_width >> shift.horizontal,
                            },
                            output: ResidentF32Plane {
                                storage: resident_binding(&upsampled[channel])?,
                                width: image_width,
                                height: image_height,
                                stride: padded_width,
                            },
                            shift: ResidentChromaShift {
                                horizontal: shift.horizontal != 0,
                                vertical: shift.vertical != 0,
                            },
                        },
                    )?);
                }
            }
            let restoration_source = pre_restoration_planes.as_ref().unwrap_or(resident_planes);
            let mut restoration = restoration_planes
                .as_ref()
                .map(|scratch| RestorationCursor::new(restoration_source, scratch));
            let gaborish_uniform = match (source.gaborish, restoration.as_mut()) {
                (Some(weights), Some(restoration)) => {
                    let (input_buffers, output_buffers) = restoration.advance();
                    let uniform = pipelines.gaborish.encode(
                        device,
                        commands,
                        ResidentGaborishInputs {
                            inputs: resident_image_planes(
                                input_buffers,
                                image_width,
                                image_height,
                                padded_width,
                            )?,
                            outputs: resident_image_planes(
                                output_buffers,
                                image_width,
                                image_height,
                                padded_width,
                            )?,
                            weights,
                        },
                    )?;
                    Some(uniform)
                }
                (None, _) => None,
                (Some(_), None) => unreachable!("Gaborish requires restoration scratch planes"),
            };
            let mut epf_uniforms =
                Vec::with_capacity(source.epf.as_ref().map_or(0, |plan| plan.passes.len()));
            if let Some(epf) = &source.epf {
                let restoration = restoration
                    .as_mut()
                    .unwrap_or_else(|| unreachable!("EPF requires restoration scratch planes"));
                let sigma_buffer = epf_sigma
                    .as_ref()
                    .unwrap_or_else(|| unreachable!("EPF requires a sigma plane"));
                let sigma = ResidentF32Plane {
                    storage: resident_binding(sigma_buffer)?,
                    width: blocks_x,
                    height: blocks_y,
                    stride: blocks_x,
                };
                for &parameters in &epf.passes {
                    let (input_buffers, output_buffers) = restoration.advance();
                    epf_uniforms.push(pipelines.epf.encode(
                        device,
                        commands,
                        ResidentEpfInputs {
                            inputs: resident_image_planes(
                                input_buffers,
                                image_width,
                                image_height,
                                padded_width,
                            )?,
                            outputs: resident_image_planes(
                                output_buffers,
                                image_width,
                                image_height,
                                padded_width,
                            )?,
                            sigma: jxl_wgpu::ResidentEpfSigma::Plane(sigma),
                            parameters,
                        },
                    )?);
                }
            }
            let presentation_planes = restoration
                .as_ref()
                .map_or(restoration_source, RestorationCursor::current);
            let mut frame_upsample_uniforms = Vec::new();
            if let (Some(upsampled), Some(weights)) =
                (&frame_upsample_planes, &frame_upsample_weights)
            {
                for channel in 0..3 {
                    frame_upsample_uniforms.push(pipelines.frame_upsample.encode(
                        device,
                        commands,
                        ResidentUpsampleInputs {
                            input: ResidentF32Plane {
                                storage: resident_binding(&presentation_planes[channel])?,
                                width: image_width,
                                height: image_height,
                                stride: padded_width,
                            },
                            output: ResidentF32Plane {
                                storage: resident_binding(&upsampled[channel])?,
                                width: source.packet.profile.output_width,
                                height: source.packet.profile.output_height,
                                stride: source.packet.profile.output_width,
                            },
                            weights,
                        },
                    )?);
                }
            }
            let presentation_planes = frame_upsample_planes
                .as_ref()
                .unwrap_or(presentation_planes);
            let presentation_shifts = if pre_restoration_planes.is_some()
                || restoration.is_some()
                || frame_upsample_planes.is_some()
            {
                [crate::vardct_frontend::VarDctChannelShift::default(); 3]
            } else {
                source.packet.profile.channel_shifts
            };
            if let crate::color_output::ColorOutputTransform::Ycbcr { channel_shifts } =
                &mut config.transform
            {
                // Plane geometry and color conversion must agree, including when noise alone
                // requires expanded components or a zero noise model elides that expansion.
                *channel_shifts = presentation_shifts;
            }
            let output_width = source.packet.profile.output_width;
            let output_height = source.packet.profile.output_height;
            let presentation_stride = if frame_upsample_planes.is_some() {
                output_width
            } else {
                padded_width
            };
            let presentation_geometry = presentation_shifts.map(|shift| {
                shift.shifted_extent(output_width, output_height).ok_or(
                    VarDctDecodeError::ArithmeticOverflow {
                        field: "presentation channel extent",
                    },
                )
            });
            let [geometry_x, geometry_y, geometry_b] = presentation_geometry;
            let presentation_geometry = [geometry_x?, geometry_y?, geometry_b?];
            let presentation_strides = presentation_shifts.map(|shift| {
                presentation_stride.checked_shr(shift.horizontal).ok_or(
                    VarDctDecodeError::ArithmeticOverflow {
                        field: "presentation channel stride",
                    },
                )
            });
            let [stride_x, stride_y, stride_b] = presentation_strides;
            let presentation_strides = [stride_x?, stride_y?, stride_b?];
            let noise = source.noise.as_ref().map(|plan| plan.allocate(device));
            let noise_uniform = if let (Some(plan), Some(scratch)) = (&source.noise, &noise) {
                let pipeline = pipelines
                    .noise
                    .get_or_init(|| jxl_wgpu::ResidentNoisePipeline::new(device))
                    .as_ref()
                    .map_err(Clone::clone)?;
                Some(pipeline.encode(
                    device,
                    commands,
                    jxl_wgpu::ResidentNoiseInputs {
                        plan,
                        scratch,
                        planes: resident_image_planes(
                            presentation_planes,
                            output_width,
                            output_height,
                            presentation_stride,
                        )?,
                    },
                )?)
            } else {
                None
            };
            let output_scratch = pipelines.output.encode(
                device,
                commands,
                ColorOutputInputs {
                    alpha: if source.surface.is_some() {
                        None
                    } else if let (Some(plan), Some(buffers)) =
                        (&source.extra_render, &rendered_extra)
                    {
                        let plane = plan.planes()[0];
                        Some(crate::color_output::ColorOutputAlpha {
                            domain: crate::ModularSampleDomain::DecodedF32,
                            storage: resident_binding(&buffers.output)?,
                            width: plane.layout.width,
                            height: plane.layout.height,
                            stride: plane.layout.row_stride_words,
                            word_offset: plane.layout.word_offset,
                            sample_bit_depth: plane.encoding.depth(),
                        })
                    } else {
                        source
                            .extra_planes
                            .first()
                            .map(super::super::staging::ResidentModularPlane::alpha_binding)
                            .transpose()?
                    },
                    planes: [
                        ColorOutputPlane {
                            storage: resident_binding(&presentation_planes[0])?,
                            width: presentation_geometry[0][0],
                            height: presentation_geometry[0][1],
                            stride: presentation_strides[0],
                        },
                        ColorOutputPlane {
                            storage: resident_binding(&presentation_planes[1])?,
                            width: presentation_geometry[1][0],
                            height: presentation_geometry[1][1],
                            stride: presentation_strides[1],
                        },
                        ColorOutputPlane {
                            storage: resident_binding(&presentation_planes[2])?,
                            width: presentation_geometry[2][0],
                            height: presentation_geometry[2][1],
                            stride: presentation_strides[2],
                        },
                    ],
                    output: resident_binding(output)?,
                    layout: &source.layout,
                    config,
                },
            )?;
            debug_assert_eq!(output_scratch.plan, plan);
            // An LF slot contains the complete pre-color-transform image, including restoration
            // and frame upsampling. Retain only these final allocations after validation.
            let lf_planes = if source.packet.profile.lf_level != 0 {
                let mut tracked = |buffer: &wgpu::Buffer| {
                    let permit = transient_permit
                        .split_off(buffer.size())
                        .map_err(ProgressiveDcGpuError::from)?;
                    Ok::<_, VarDctDecodeError>(GpuBufferLease::from_tracked(buffer.clone(), permit))
                };
                Some(ProgressiveDcXybPlanes::from_leases(
                    [
                        tracked(&presentation_planes[0])?,
                        tracked(&presentation_planes[1])?,
                        tracked(&presentation_planes[2])?,
                    ],
                    output_width,
                    output_height,
                    presentation_stride,
                )?)
            } else {
                None
            };
            let post_transform_buffers = PostTransformJobBuffers {
                _noise: noise,
                _noise_uniform: noise_uniform,
                _restoration_planes: restoration_planes,
                _pre_restoration_planes: pre_restoration_planes,
                _pre_restoration_uniforms: pre_restoration_uniforms,
                _gaborish_uniform: gaborish_uniform,
                _epf_sigma: epf_sigma,
                _epf_sigma_uniforms: epf_sigma_uniforms,
                _epf_uniforms: epf_uniforms,
                _frame_upsample_planes: frame_upsample_planes,
                _frame_upsample_weights: frame_upsample_weights,
                _frame_upsample_uniforms: frame_upsample_uniforms,
            };
            (
                FrameOutputScratch::Color {
                    _scratch: output_scratch,
                },
                post_transform_buffers,
                lf_planes,
            )
        }
        VarDctFrameOutput::Extra { index, plan } => {
            let extra =
                source
                    .extra_planes
                    .first()
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "scalar output lacks its selected resident extra plane",
                    })?;
            if plan.config.encoding != extra.encoding || index != extra.index {
                return Err(VarDctDecodeError::EntropyWindowContract {
                    detail: "scalar output precision differs from the selected extra plane",
                });
            }
            let (plane, domain, arena) =
                if let (Some(plan), Some(buffers)) = (&source.extra_render, &rendered_extra) {
                    (
                        plan.planes()[0].layout,
                        crate::ModularSampleDomain::DecodedF32,
                        &buffers.output,
                    )
                } else {
                    (
                        extra.plane,
                        crate::ModularSampleDomain::Encoded,
                        extra.arena.as_wgpu_buffer(),
                    )
                };
            let scratch = pipelines.scalar_output.encode(
                device,
                commands,
                plan,
                crate::modular_scalar_output::ModularScalarOutputInputs {
                    plane,
                    domain,
                    arena: resident_binding(arena)?,
                    output: resident_binding(output)?,
                },
            )?;
            (
                FrameOutputScratch::Extra { scratch },
                PostTransformJobBuffers::default(),
                None,
            )
        }
    };
    if let Some(surface) = &source.surface
        && !surface.extras.is_empty()
    {
        let plan =
            source
                .extra_render
                .as_ref()
                .ok_or(VarDctDecodeError::EntropyWindowContract {
                    detail: "frame surface lacks normalized extra planes",
                })?;
        let buffers = rendered_extra
            .as_ref()
            .ok_or(VarDctDecodeError::EntropyWindowContract {
                detail: "frame surface lacks normalized extra storage",
            })?;
        if plan.planes().len() != surface.extras.len() {
            return Err(VarDctDecodeError::EntropyWindowContract {
                detail: "frame surface extra plane count changed",
            });
        }
        for (plane, layout) in plan.planes().iter().zip(&surface.extras) {
            let plane = plane.layout;
            commands.copy_buffer_to_buffer(
                &buffers.output,
                u64::from(plane.word_offset) * 4,
                output,
                layout.planes[0].offset,
                u64::from(plane.width) * u64::from(plane.height) * 4,
            );
        }
    }
    Ok(FrameRenderResult {
        output_scratch,
        post_transform_buffers,
        lf_planes,
        resident_scratch,
    })
}
