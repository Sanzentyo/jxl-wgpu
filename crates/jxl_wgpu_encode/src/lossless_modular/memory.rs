use super::grid::LosslessModularGroupGrid;
use super::types::LosslessModularFormat;
use crate::EncodeError;

/// Checked memory accounting for one concrete Modular submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularMemoryPlan {
    pub group_grid: LosslessModularGroupGrid,
    pub format: LosslessModularFormat,
    /// Valid low bits in every unsigned integer component (`1..=16`).
    pub bits_per_sample: u8,
    /// Native storage bytes occupied by every component (`1` or `2`).
    pub bytes_per_sample: u8,
    /// Number of independently tokenized Modular channels (1, 3, or 4).
    pub channel_count: u32,
    /// Full source byte range addressed by the logical frame.
    pub source_binding_bytes: u64,
    /// Largest source window bound for any one streamed GPU batch.
    pub peak_source_binding_bytes: u64,
    /// Largest parameter allocation used by one streamed GPU batch.
    pub parameter_storage_bytes: u64,
    /// Largest artifact allocation used by one streamed GPU batch.
    pub artifact_storage_bytes: u64,
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
    pub streaming: bool,
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
