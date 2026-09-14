//! A selected image color connection shared by reconstruction and presentation.

use std::sync::{Arc, Mutex};

use jxl_gpu_protocol::icc::IccTransform;
use jxl_wgpu::{
    MemoryPermit, ResidentIccDispatch, ResidentIccInputs, ResidentIccMemoryPlan,
    ResidentIccPipeline, ResidentIccProgram, ResidentStorageBinding, WgpuBackend,
};

use crate::{Error, Result};

/// A planning-time registry. Executable connections retain shared ownership after planning.
#[derive(Default)]
pub(super) struct Transforms {
    connections: Vec<Arc<Transform>>,
}

impl Transforms {
    pub(super) fn select(
        &mut self,
        backend: &WgpuBackend,
        selected: IccTransform,
    ) -> Result<Arc<Transform>> {
        if let Some(existing) = self
            .connections
            .iter()
            .find(|entry| entry.selected == selected)
        {
            return Ok(Arc::clone(existing));
        }
        let connection = Arc::new(Transform::new(backend, selected)?);
        self.connections.push(Arc::clone(&connection));
        Ok(connection)
    }
}

#[derive(Debug)]
pub(super) struct Transform {
    selected: IccTransform,
    pub(super) memory: ResidentIccMemoryPlan,
    pipeline: ResidentIccPipeline,
    // Failed admissions leave this empty. Each submitted use retains its own program Arc.
    pub(super) uploaded: Mutex<Option<Arc<Program>>>,
}

#[derive(Debug)]
pub(super) struct Program {
    resident: ResidentIccProgram,
    _permit: MemoryPermit,
}

pub(super) struct ColorBinding<'a> {
    pub(super) storage: ResidentStorageBinding<'a>,
    pub(super) layout: &'a crate::frame_surface::FrameSurfaceLayout,
    pub(super) encoding: &'a crate::frame_surface::FrameSurfaceEncoding,
}

impl Transform {
    fn new(backend: &WgpuBackend, selected: IccTransform) -> Result<Self> {
        Ok(Self {
            memory: ResidentIccMemoryPlan::new(&selected, &backend.device().limits())?,
            pipeline: ResidentIccPipeline::new(backend.device())?,
            selected,
            uploaded: Mutex::new(None),
        })
    }

    pub(super) fn resident(&self, backend: &WgpuBackend) -> Result<Arc<Program>> {
        let mut cached = super::lock(&self.uploaded);
        if let Some(program) = &*cached {
            return Ok(Arc::clone(program));
        }
        let permit = backend
            .transient_memory_budget()
            .try_reserve(self.memory.program_bytes)?;
        let program = Arc::new(Program {
            resident: ResidentIccProgram::new(backend.device(), &self.selected)?,
            _permit: permit,
        });
        *cached = Some(Arc::clone(&program));
        Ok(program)
    }

    pub(super) fn encode(
        &self,
        backend: &WgpuBackend,
        encoder: &mut wgpu::CommandEncoder,
        program: &Program,
        source: ColorBinding<'_>,
        target: ColorBinding<'_>,
    ) -> Result<ResidentIccDispatch> {
        if source.layout.color.extent != target.layout.color.extent {
            return Err(Error::EngineContract("ICC connection extent mismatch"));
        }
        Ok(self.pipeline.encode(
            backend.device(),
            encoder,
            &program.resident,
            ResidentIccInputs {
                input: source.storage,
                output: target.storage,
                extent: source.layout.color.extent,
                input_planes: &source.layout.icc_planes(source.encoding)?,
                output_planes: &target.layout.icc_planes(target.encoding)?,
                input_encoding: source.encoding.icc_sample_encoding(),
                output_encoding: target.encoding.icc_sample_encoding(),
            },
        )?)
    }
}
