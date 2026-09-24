use super::grid::LosslessModularGroupGrid;
use super::types::LosslessModularFormat;
use crate::EncodeError;

/// Checked memory accounting for one concrete Modular submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularMemoryPlan {
    pub group_grid: LosslessModularGroupGrid,
    pub format: LosslessModularFormat,
    /// Valid bits in each integer component, or total IEEE floating-point width.
    pub bits_per_sample: u8,
    /// Zero for integer samples, five for binary16, or eight for binary32.
    pub exponent_bits_per_sample: u8,
    /// Largest storage word containing a component (`1`, `2`, `3`, or `4` bytes).
    pub bytes_per_sample: u8,
    /// Maximum independently tokenized channels in any group after all local transforms.
    /// Single-pixel edge axes are skipped, so individual groups may contain fewer channels.
    pub channel_count: u32,
    /// Union of the full source plane binding ranges, excluding gaps between planes.
    pub source_binding_bytes: u64,
    /// Largest union of source binding ranges in one GPU batch. Each plane has its own binding.
    pub peak_source_binding_bytes: u64,
    /// Largest parameter allocation used by one streamed GPU batch.
    pub parameter_storage_bytes: u64,
    /// Largest artifact allocation used by one streamed GPU batch.
    pub artifact_storage_bytes: u64,
    /// Peak live Weighted predictor row state (20 bytes per column and channel), already
    /// included in `artifact_storage_bytes`. Zero for the other thirteen predictors.
    pub weighted_predictor_scratch_bytes: u64,
    /// Peak residual words, hash-chain links and bucket heads for greedy LZ77. Already
    /// included in `artifact_storage_bytes`; zero for the default zero-run policy.
    pub lz77_scratch_bytes: u64,
    /// Peak palette dictionary/count/hash, delta residual and delta Weighted row storage,
    /// already included in artifact bytes.
    pub palette_scratch_bytes: u64,
    /// Peak explicit transform operation tables and live sample arenas, included in artifact bytes.
    pub transform_scratch_bytes: u64,
    /// Peak GPU ANS output capacity, already included in artifact bytes; zero for Prefix.
    /// Immutable ANS tables and descriptors are included in `parameter_storage_bytes`.
    pub ans_output_bytes: u64,
    /// Peak GPU hybrid-uint histogram storage, included in artifact bytes; zero for Prefix.
    pub hybrid_histogram_bytes: u64,
    /// Sum of the worst-case artifact ranges across every batch. This is diagnostic only; the
    /// encoder never allocates the sum as one GPU buffer.
    pub total_artifact_bytes: u64,
    /// Separate copy destination required before mapping. Zero when the device can map the
    /// primary storage buffer directly.
    pub readback_bytes: u64,
    pub direct_readback: bool,
    /// Artifact batches needed to cover the frame.
    pub batch_count: u32,
    /// Actual `wgpu::Queue::submit` calls made by this job. Resident jobs submit once; streamed
    /// jobs submit every batch once for histogram aggregation and once for serialization.
    pub gpu_submission_count: u32,
    /// Two-pass scheduling, selected for multiple batches or GPU ANS even with one batch.
    pub streaming: bool,
    /// Caller-owned ICC bytes retained by the source/descriptor; zero for enumerated color.
    pub icc_profile_bytes: u64,
    /// Shared-budget reservation for the ICC image header and a temporary assembly copy.
    /// Included in `owned_bytes_per_job` for the complete encoder; the frame backend sets zero.
    pub icc_storage_bytes: u64,
    pub owned_bytes_per_job: u64,
    pub addressed_bytes_per_job: u64,
}

/// Total memory exposure for a caller-selected maximum number of in-flight jobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularInFlightMemory {
    pub max_in_flight_jobs: u32,
    pub total_owned_bytes: u64,
    pub total_addressed_bytes: u64,
}

/// Device limits that bound concrete Modular source and artifact bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularMemoryLimits {
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub max_compute_workgroups_per_dimension: u32,
}

impl LosslessModularMemoryPlan {
    #[must_use]
    pub const fn sample_bit_depth(&self) -> jxl_gpu_bitstream::SampleBitDepth {
        super::types::modular_sample_depth(self.bits_per_sample, self.exponent_bits_per_sample)
    }

    pub fn for_in_flight(
        self,
        max_in_flight_jobs: u32,
    ) -> Result<LosslessModularInFlightMemory, EncodeError> {
        if max_in_flight_jobs == 0 {
            return Err(EncodeError::InvalidConfiguration(
                "max in-flight job count must be non-zero",
            ));
        }
        let jobs = u64::from(max_in_flight_jobs);
        let total_owned_bytes =
            self.owned_bytes_per_job
                .checked_mul(jobs)
                .ok_or(EncodeError::InvalidConfiguration(
                    "in-flight encoder memory size overflow",
                ))?;
        let total_addressed_bytes = self.addressed_bytes_per_job.checked_mul(jobs).ok_or(
            EncodeError::InvalidConfiguration("in-flight encoder memory size overflow"),
        )?;
        Ok(LosslessModularInFlightMemory {
            max_in_flight_jobs,
            total_owned_bytes,
            total_addressed_bytes,
        })
    }
}
pub(super) fn align_up(value: u64, alignment: u64) -> Option<u64> {
    let adjustment = alignment.checked_sub(1)?;
    value
        .checked_add(adjustment)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}
pub(super) fn event_capacity(pixel_count: usize) -> Result<usize, EncodeError> {
    pixel_count
        .checked_add(pixel_count.div_ceil(8))
        .and_then(|value| value.checked_add(1))
        .ok_or(EncodeError::InvalidSource("event buffer size overflow"))
}
