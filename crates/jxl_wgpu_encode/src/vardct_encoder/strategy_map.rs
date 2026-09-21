//! Validated transform placement and bounded metadata for batched GPU execution.

use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{ForwardVarDctMemoryPlan, ForwardVarDctTask};

use super::entropy::{HfEntropyPlan, fixed_prefix_code};
use super::types::{
    ArtifactLayout, TiledVarDctGrid, VarDctFrameLayout, VarDctStrategy, VarDctTopology,
    VarDctTransformMemoryPlan,
};
use super::{VarDctCoefficientOrders, VarDctHfMultiplier, VarDctQuantization};
use crate::EncodeError;

/// A transform's upper-left corner, measured in 8×8 blocks of the padded image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctTransform {
    pub block_x: u32,
    pub block_y: u32,
    pub strategy: VarDctStrategy,
    /// Uses the frame's default HF multiplier when absent.
    pub hf_multiplier: Option<VarDctHfMultiplier>,
}

impl VarDctTransform {
    #[must_use]
    pub const fn new(block_x: u32, block_y: u32, strategy: VarDctStrategy) -> Self {
        Self {
            block_x,
            block_y,
            strategy,
            hf_multiplier: None,
        }
    }

    #[must_use]
    pub const fn with_hf_multiplier(mut self, multiplier: VarDctHfMultiplier) -> Self {
        self.hf_multiplier = Some(multiplier);
        self
    }
}

/// An exact covering of the image's 8×8 block grid with standard VarDCT transforms.
///
/// Placements are canonicalized into raster order. Rectangles must not overlap,
/// leave holes, extend past the padded grid, or cross a 256-pixel AC-group boundary.
/// Partial edge blocks replicate the final source row/column on the GPU. This is
/// caller-supplied metadata; it does not perform content-adaptive strategy selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VarDctStrategyMap {
    extent: Extent2d,
    transforms: Vec<VarDctTransform>,
    pub(super) block_map: Vec<u32>,
}

impl VarDctStrategyMap {
    pub fn new(
        width: u32,
        height: u32,
        mut transforms: Vec<VarDctTransform>,
    ) -> Result<Self, EncodeError> {
        let grid = TiledVarDctGrid::new(width, height)?;
        if transforms.is_empty() || transforms.len() > grid.block_count()? as usize {
            return Err(EncodeError::InvalidConfiguration(
                "VarDCT strategy map has an invalid transform count",
            ));
        }
        let mut block_map = vec![u32::MAX; grid.block_count()? as usize];
        transforms.sort_unstable_by_key(|task| (task.block_y, task.block_x));
        for task in &transforms {
            let lf = task.strategy.lf_extent();
            if task.block_x >= grid.block_columns
                || task.block_y >= grid.block_rows
                || lf.width > grid.block_columns - task.block_x
                || lf.height > grid.block_rows - task.block_y
            {
                return Err(EncodeError::InvalidConfiguration(
                    "VarDCT transform exceeds the padded image grid",
                ));
            }
            if lf.width > 32 - task.block_x % 32 || lf.height > 32 - task.block_y % 32 {
                return Err(EncodeError::InvalidConfiguration(
                    "VarDCT transform crosses an AC-group boundary",
                ));
            }
            for y in 0..lf.height {
                for x in 0..lf.width {
                    let slot = &mut block_map
                        [((task.block_y + y) * grid.block_columns + task.block_x + x) as usize];
                    if *slot != u32::MAX {
                        return Err(EncodeError::InvalidConfiguration(
                            "VarDCT transforms overlap",
                        ));
                    }
                    *slot = u32::from(task.strategy.codestream_id())
                        | (u32::from(x == 0 && y == 0) << 8);
                }
            }
        }
        if block_map.contains(&u32::MAX) {
            return Err(EncodeError::InvalidConfiguration(
                "VarDCT strategy map leaves uncovered blocks",
            ));
        }
        Ok(Self {
            extent: Extent2d { width, height },
            transforms,
            block_map,
        })
    }

    #[must_use]
    pub const fn extent(&self) -> Extent2d {
        self.extent
    }

    #[must_use]
    pub fn transforms(&self) -> &[VarDctTransform] {
        &self.transforms
    }

    pub(super) fn frame(&self) -> Result<VarDctFrameLayout, EncodeError> {
        let mut frame = VarDctFrameLayout::tiled_dct8(self.extent.width, self.extent.height)?;
        frame.topology = VarDctTopology::StrategyMap;
        Ok(frame)
    }
}

/// Scalar offsets and quantization metadata; exactly 44 bytes in WGSL.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct TransformTask {
    pub block_x: u32,
    pub block_y: u32,
    pub coefficient_offset: u32,
    pub lf_offset: u32,
    pub width: u32,
    pub height: u32,
    pub metadata_offset: u32,
    pub ac_word_offset: u32,
    pub ac_word_capacity: u32,
    pub strategy: u32,
    pub hf_multiplier: u32,
}

#[derive(Debug)]
pub(super) struct StrategyBatch {
    pub strategy: VarDctStrategy,
    pub tasks: Vec<ForwardVarDctTask>,
    pub memory: ForwardVarDctMemoryPlan,
}

#[derive(Debug)]
pub(super) struct TransformPlan {
    pub map: VarDctStrategyMap,
    pub tasks: Vec<TransformTask>,
    pub batches: Vec<StrategyBatch>,
    pub metadata: Vec<[u32; 6]>,
    pub memory: VarDctTransformMemoryPlan,
    pub ac_words: u32,
    pub ac_groups: Vec<Vec<usize>>,
    pub lf_groups: Vec<Vec<usize>>,
}

impl TransformPlan {
    pub fn new(
        map: VarDctStrategyMap,
        quantization: VarDctQuantization,
        orders: &VarDctCoefficientOrders,
    ) -> Result<Self, EncodeError> {
        let code = fixed_prefix_code()?;
        let mut metadata = Vec::new();
        let mut batches = Vec::new();
        let mut offsets = [0; 27];
        let mut capacities = [0; 27];
        for strategy in VarDctStrategy::ALL {
            let count = map
                .transforms
                .iter()
                .filter(|task| task.strategy == strategy)
                .count() as u32;
            if count == 0 {
                continue;
            }
            let id = usize::from(strategy.codestream_id());
            offsets[id] = metadata.len() as u32;
            capacities[id] = ArtifactLayout::new(strategy, &code)?.ac_words_per_block;
            metadata.extend(
                strategy
                    .default_dequant_matrix()
                    .scales
                    .into_iter()
                    .zip(orders.indices(strategy))
                    .map(|([x, y, b], [ox, oy, ob])| {
                        [x.to_bits(), y.to_bits(), b.to_bits(), ox, oy, ob]
                    }),
            );
            batches.push(StrategyBatch {
                strategy,
                tasks: Vec::with_capacity(count as usize),
                memory: ForwardVarDctMemoryPlan::for_batch(strategy, count)?,
            });
        }
        let mut tasks = Vec::with_capacity(map.transforms.len());
        let mut coefficient_offset = 0;
        let mut lf_offset = 0;
        let mut ac_words = 0u32;
        let padded_width = map.extent.width.div_ceil(8) * 8;
        let canvas_area = padded_width * map.extent.height.div_ceil(8) * 8;
        for placement in &map.transforms {
            let strategy = placement.strategy;
            let id = usize::from(strategy.codestream_id());
            let extent = strategy.pixel_extent();
            let area = extent.width * extent.height;
            tasks.push(TransformTask {
                block_x: placement.block_x,
                block_y: placement.block_y,
                coefficient_offset,
                lf_offset,
                width: extent.width,
                height: extent.height,
                metadata_offset: offsets[id],
                ac_word_offset: ac_words,
                ac_word_capacity: capacities[id],
                strategy: id as u32,
                hf_multiplier: placement
                    .hf_multiplier
                    .unwrap_or(quantization.hf_multiplier())
                    .get(),
            });
            batches
                .iter_mut()
                .find(|batch| batch.strategy == strategy)
                .expect("populated strategy batch")
                .tasks
                .push(ForwardVarDctTask {
                    origins: std::array::from_fn(|channel| {
                        channel as u32 * canvas_area
                            + placement.block_y * 8 * padded_width
                            + placement.block_x * 8
                    }),
                    coefficient_offset,
                    lf_offset,
                });
            coefficient_offset += area * 3;
            lf_offset += area / 64 * 3;
            ac_words =
                ac_words
                    .checked_add(capacities[id])
                    .ok_or(EncodeError::InvalidConfiguration(
                        "VarDCT AC arena overflow",
                    ))?;
        }
        let mut forward = batches[0].memory;
        for batch in &batches[1..] {
            forward.parameter_bytes += batch.memory.parameter_bytes;
            forward.basis_bytes += batch.memory.basis_bytes;
            forward.horizontal_bytes += batch.memory.horizontal_bytes;
            forward.task_bytes += batch.memory.task_bytes;
            forward.transient_bytes += batch.memory.transient_bytes;
            forward.coefficient_bytes += batch.memory.coefficient_bytes;
            forward.lf_bytes += batch.memory.lf_bytes;
        }
        let mut memory = VarDctTransformMemoryPlan::new(map.transforms[0].strategy);
        memory.forward = forward;
        memory.xyb_bytes = u64::from(coefficient_offset) * 4;
        memory.coefficient_bytes = memory.xyb_bytes;
        memory.quantized_bytes = memory.xyb_bytes;
        memory.lf_bytes = u64::from(lf_offset) * 4;
        memory.quantization_metadata_bytes = metadata.len() as u64 * 24;
        memory.task_metadata_bytes =
            tasks.len() as u64 * std::mem::size_of::<TransformTask>() as u64;
        memory.total_bytes = forward.transient_bytes
            + 3 * memory.xyb_bytes
            + memory.lf_bytes
            + memory.quantization_metadata_bytes
            + memory.task_metadata_bytes;
        let frame = map.frame()?;
        let mut ac_groups = vec![Vec::new(); frame.ac_group_count()? as usize];
        let mut lf_groups = vec![Vec::new(); frame.lf_group_count()? as usize];
        for (index, task) in tasks.iter().enumerate() {
            ac_groups[(task.block_y / 32 * frame.ac_groups_x + task.block_x / 32) as usize]
                .push(index);
            lf_groups[(task.block_y / 256 * frame.lf_groups_x + task.block_x / 256) as usize]
                .push(index);
        }
        Ok(Self {
            map,
            tasks,
            batches,
            metadata,
            memory,
            ac_words,
            ac_groups,
            lf_groups,
        })
    }

    pub fn artifact_layout(
        &self,
        frame: VarDctFrameLayout,
        code: &super::entropy::VarDctPrefixCode,
    ) -> Result<ArtifactLayout, EncodeError> {
        ArtifactLayout::for_strategy_map(frame, code, self.tasks.len() as u32, self.ac_words)
    }

    pub fn validate_ac(
        &self,
        words: &[u32],
        lengths: &[u32],
        entropy: &HfEntropyPlan,
    ) -> Result<(), crate::BackendError> {
        if words.len() != self.ac_words as usize || lengths.len() != self.tasks.len() {
            return Err(crate::BackendError::InvalidArtifact(
                "VarDCT mapped AC allocation mismatch",
            ));
        }
        for (task, length) in self.tasks.iter().zip(lengths) {
            let start = task.ac_word_offset as usize;
            let end = start + task.ac_word_capacity as usize;
            super::ac::validate_transform_fragments(
                &words[start..end],
                std::slice::from_ref(length),
                task.ac_word_capacity,
                task.width * task.height / 64 * 63,
                entropy,
            )?;
        }
        Ok(())
    }
}
