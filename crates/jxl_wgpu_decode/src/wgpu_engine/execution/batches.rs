//! One bounded entropy batch, shared by eager final decoding and pass continuations.
use super::*;

pub(super) struct BatchEncoder<'a> {
    pub backend: &'a WgpuBackend,
    pub source: &'a DecodeSource,
    pub pipelines: &'a SubmitPipelines,
    pub lifetime: &'a DecodeJobLifetime,
    pub stream: &'a wgpu::Buffer,
    pub binding: &'a wgpu::BindGroup,
    pub global_binding: Option<&'a wgpu::BindGroup>,
    pub stream_upload: &'a mut [u8],
}

impl BatchEncoder<'_> {
    pub(super) fn global(
        &mut self,
        batch_index: usize,
        commands: &mut wgpu::CommandEncoder,
    ) -> Result<()> {
        let backend = self.backend;
        let source = self.source;
        let pipelines = self.pipelines;
        let lifetime = self.lifetime;
        let stream = self.stream;
        let binding = self.binding;
        let stream_upload = &mut *self.stream_upload;
        let global_binding = self.global_binding;
        let batch = source
            .dispatch_layout
            .global_streams
            .batch(batch_index)
            .ok_or(Error::EngineContract("global entropy batch is missing"))?;
        let global_record_index = source.profile.entropy_groups.len();
        let global_record_index_u32 = u32::try_from(global_record_index)
            .map_err(|_| Error::backend("DC-global status index exceeds WGSL u32"))?;
        let global_params_offset = u64::try_from(global_record_index)
            .ok()
            .and_then(|index| index.checked_mul(source.dispatch_layout.params_stride))
            .ok_or_else(|| Error::backend("DC-global parameter offset overflow"))?;
        stream_upload.fill(0);
        let [segment] = batch.segments() else {
            return Err(Error::EngineContract(
                "one DC-global entropy batch must contain exactly one segment",
            ));
        };
        let segment = *segment;
        copy_stream_segment(source, segment, stream_upload, "DC-global")?;
        let params = build_global_params(segment, global_record_index_u32, source)?;
        backend.queue().write_buffer(
            lifetime._params.buffer(),
            global_params_offset,
            bytemuck::bytes_of(&params),
        );
        backend.queue().write_buffer(stream, 0, stream_upload);
        let control = DispatchControl {
            first_group: global_record_index_u32,
            group_count: 1,
            lane_stride_words: u32::try_from(source.dispatch_layout.reconstruction_lane_stride / 4)
                .map_err(|_| Error::backend("reconstruction lane stride exceeds WGSL u32"))?,
            _padding: 0,
        };
        backend.queue().write_buffer(
            lifetime._dispatch_control.buffer(),
            0,
            bytemuck::bytes_of(&control),
        );
        if batch_index == 0 {
            commands.clear_buffer(lifetime._reconstructed.buffer(), 0, None);
            if let Some(frame_arena) = &lifetime._frame_arena {
                commands.clear_buffer(frame_arena.buffer(), 0, None);
            }
            commands.clear_buffer(lifetime.output.as_wgpu_buffer(), 0, None);
            if let Some(dummy) = &lifetime._native_f64_dummy_words {
                commands.clear_buffer(dummy.buffer(), 0, None);
            }
            commands.clear_buffer(lifetime._status.buffer(), 0, None);
        }
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jxl-wgpu DC-global Modular entropy reconstruction"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.decode);
            pass.set_bind_group(0, global_binding.unwrap_or(binding), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        Ok(())
    }

    pub(super) fn group(
        &mut self,
        batch_index: usize,
        commands: &mut wgpu::CommandEncoder,
    ) -> Result<Vec<wgpu::Buffer>> {
        let backend = self.backend;
        let device = backend.device();
        let source = self.source;
        let pipelines = self.pipelines;
        let lifetime = self.lifetime;
        let stream = self.stream;
        let binding = self.binding;
        let stream_upload = &mut *self.stream_upload;
        let has_global_stream = source.dispatch_layout.global_streams.batch_count() != 0;
        let batch = source
            .dispatch_layout
            .streams
            .batch(batch_index)
            .ok_or(Error::EngineContract("group entropy batch is missing"))?;
        stream_upload.fill(0);
        for &segment in batch.segments() {
            copy_stream_segment(source, segment, stream_upload, "group")?;

            let group = source
                .profile
                .entropy_groups
                .get(segment.group_index)
                .copied()
                .ok_or_else(|| Error::backend("stream segment group index is invalid"))?;
            let status_index = u32::try_from(segment.group_index)
                .map_err(|_| Error::backend("group status index exceeds WGSL u32"))?;
            let params = build_params(
                group,
                segment.group_index,
                segment,
                status_index,
                source,
                source.dispatch_layout.reconstruction_specialization,
                segment.group_index == 0,
            )?;
            let params_offset = u64::try_from(segment.group_index)
                .ok()
                .and_then(|index| index.checked_mul(source.dispatch_layout.params_stride))
                .ok_or_else(|| Error::backend("group parameter offset overflow"))?;
            let params_end = params_offset
                .checked_add(std::mem::size_of::<ShaderParams>() as u64)
                .ok_or_else(|| Error::backend("group parameter range overflow"))?;
            if params_end > source.dispatch_layout.params_bytes {
                return Err(Error::backend("group parameter buffer is truncated"));
            }
            backend.queue().write_buffer(
                lifetime._params.buffer(),
                params_offset,
                bytemuck::bytes_of(&params),
            );
        }
        backend.queue().write_buffer(stream, 0, stream_upload);
        let control = DispatchControl {
            first_group: u32::try_from(batch.first_group())
                .map_err(|_| Error::backend("batch group index exceeds WGSL u32"))?,
            group_count: u32::try_from(batch.group_count())
                .map_err(|_| Error::backend("batch group count exceeds WGSL u32"))?,
            lane_stride_words: u32::try_from(source.dispatch_layout.reconstruction_lane_stride / 4)
                .map_err(|_| Error::backend("reconstruction lane stride exceeds WGSL u32"))?,
            _padding: 0,
        };
        backend.queue().write_buffer(
            lifetime._dispatch_control.buffer(),
            0,
            bytemuck::bytes_of(&control),
        );
        if batch_index == 0 {
            commands.clear_buffer(lifetime._reconstructed.buffer(), 0, None);
            if !has_global_stream {
                if let Some(frame_arena) = &lifetime._frame_arena {
                    commands.clear_buffer(frame_arena.buffer(), 0, None);
                }
                commands.clear_buffer(lifetime.output.as_wgpu_buffer(), 0, None);
                if let Some(dummy) = &lifetime._native_f64_dummy_words {
                    commands.clear_buffer(dummy.buffer(), 0, None);
                }
                commands.clear_buffer(lifetime._status.buffer(), 0, None);
            }
        }
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jxl-wgpu generic Modular entropy and MA reconstruction"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.decode);
            pass.set_bind_group(0, binding, &[]);
            pass.dispatch_workgroups(
                control
                    .group_count
                    .div_ceil(source.dispatch_layout.group_workgroup_size),
                1,
                1,
            );
        }
        let mut inverse_uniforms = Vec::new();
        if !source.channel_layout_offsets.is_empty() {
            let inverse = pipelines.inverse.as_deref().ok_or(Error::EngineContract(
                "descriptor reconstruction is missing resident inverse pipelines",
            ))?;
            for &segment in batch.segments() {
                if segment.flags & GroupStreamSegment::FINAL == 0 {
                    continue;
                }
                let lane_index = segment
                    .group_index
                    .checked_sub(batch.first_group())
                    .filter(|lane| *lane < batch.group_count())
                    .ok_or_else(|| {
                        Error::backend("final Modular group lane is outside its batch")
                    })?;
                inverse_uniforms.extend(encode_modular_inverse(
                    device,
                    commands,
                    source,
                    lifetime._reconstructed.buffer(),
                    inverse,
                    segment.group_index,
                    lane_index,
                )?);
                if source.profile.resident_frame_plan.is_some() {
                    encode_subimage_plane_copies(
                        commands,
                        source,
                        lifetime._reconstructed.buffer(),
                        lifetime
                            ._frame_arena
                            .as_ref()
                            .ok_or(Error::EngineContract(
                                "Modular subimage assembly is missing its frame arena",
                            ))?
                            .buffer(),
                        segment.group_index,
                        lane_index,
                    )?;
                } else if source.profile.progressive_dc.is_none() {
                    inverse_uniforms.extend(encode_modular_finalize(
                        device,
                        commands,
                        source,
                        lifetime,
                        inverse,
                        segment.group_index,
                        lane_index,
                    )?);
                }
            }
        }
        Ok(inverse_uniforms)
    }
}
