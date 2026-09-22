use jxl_gpu_bitstream::{
    ContainerStreamEvent, FrameIndexCollector, FrameIndexCollectorStats, FrameIndexLimits,
};

use super::{BoundFrameIndex, FrameSeekError, FrameSeekLimits, GpuSeekSession, open_seek_plan};
use crate::{
    Error, GpuDecodeSession, GpuDecodeStream, GpuDecodeStreamStats, GpuDecoder, GpuOutputRequest,
    GpuSubmissionEngine, ImageSelection, Result,
};

impl<E: GpuSubmissionEngine> GpuDecoder<E> {
    /// Collects a bounded index alongside shared codestream spans, without joining complete input.
    /// Feed events from a [`jxl_gpu_bitstream::ContainerStreamScanner`] using this decoder's
    /// transport limits. The requested target is selected only after authoritative transport End.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use jxl_gpu_bitstream::ContainerStreamScanner;
    /// # use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, WgpuDecodeEngine, vardct_rgb8_format};
    /// # fn example(decoder: &GpuDecoder<WgpuDecodeEngine>, chunks: impl IntoIterator<Item = Arc<[u8]>>)
    /// # -> Result<(), Box<dyn std::error::Error>> {
    /// let request = GpuOutputRequest::color(vardct_rgb8_format())?;
    /// let mut input = decoder.stream_seek(request, Default::default())?;
    /// let mut transport = ContainerStreamScanner::new(decoder.container_stream_limits());
    /// for chunk in chunks {
    ///     for event in transport.push_chunk(chunk)? {
    ///         input.push_transport_event(&event)?;
    ///     }
    /// }
    /// for event in transport.finish_input()? {
    ///     input.push_transport_event(&event)?;
    /// }
    /// let mut seek = input.finish(2, Default::default())?;
    /// let frame = seek.next_frame()?.expect("requested presentation");
    /// assert_eq!(frame.metadata.index, 2);
    /// # Ok(()) }
    /// ```
    pub fn stream_seek(
        &self,
        request: GpuOutputRequest,
        index_limits: FrameIndexLimits,
    ) -> Result<GpuDecodeSeekStream<E>> {
        if request.image_selection() != ImageSelection::Main {
            return Err(FrameSeekError::PreviewSelection.into());
        }
        Ok(GpuDecodeSeekStream {
            input: self.stream(request)?,
            index: Some(FrameIndexCollector::new(index_limits)),
            index_limits,
        })
    }
}

/// Incremental input and index ownership for a subsequent one-presentation GPU seek.
/// Missing indexes are generated after complete header inventory. A present but invalid index
/// cannot be ignored. No GPU target session opens until transport, inventory, index and dependency
/// checks pass. Completed previews remain independently available before main input ends.
pub struct GpuDecodeSeekStream<E: GpuSubmissionEngine> {
    input: GpuDecodeStream<E>,
    index: Option<FrameIndexCollector>,
    index_limits: FrameIndexLimits,
}

impl<E: GpuSubmissionEngine> GpuDecodeSeekStream<E> {
    #[must_use]
    pub fn stats(&self) -> GpuDecodeStreamStats {
        self.input.stats()
    }

    #[must_use]
    pub fn index_stats(&self) -> FrameIndexCollectorStats {
        self.index.as_ref().map_or(
            FrameIndexCollectorStats {
                failed: true,
                ..Default::default()
            },
            FrameIndexCollector::stats,
        )
    }

    #[must_use]
    pub const fn is_ready(&self) -> bool {
        self.input.is_ready()
    }

    #[must_use]
    pub fn is_preview_ready(&self) -> bool {
        self.input.is_preview_ready()
    }

    /// Same independently validated preview operation as [`GpuDecodeStream::take_preview`].
    pub fn take_preview(
        &mut self,
        request: GpuOutputRequest,
    ) -> Result<Option<GpuDecodeSession<E::Session>>> {
        self.input.take_preview(request)
    }

    /// Input byte/span pressure consumes neither frontend and can retry the identical event.
    /// Other failures poison both frontends and retire their owned input/index data.
    pub fn push_transport_event(&mut self, event: &ContainerStreamEvent) -> Result<()> {
        if let Err(error) = self.input.push_transport_event(event) {
            if self.input.failed {
                self.index = None;
            }
            return Err(error);
        }
        let result = self
            .index
            .as_mut()
            .ok_or(Error::IncrementalInputPoisoned)?
            .push_transport_event(event)
            .map_err(FrameSeekError::from);
        if let Err(error) = result {
            self.index = None;
            return self.input.fail(error.into());
        }
        Ok(())
    }

    /// Requires authoritative transport End, binds the index and restores the target dependency
    /// interval through the same GPU engine used by [`GpuDecoder::open_seek`]. Shared span tokens
    /// move into that session; any failed handoff releases them. This does not acquire byte ranges.
    pub fn finish(
        mut self,
        target: usize,
        limits: FrameSeekLimits,
    ) -> Result<GpuSeekSession<E::Session>> {
        let (codestream, inventory) = self.input.finish_parts()?;
        let index = self
            .index
            .take()
            .ok_or(Error::IncrementalInputPoisoned)?
            .finish()
            .map_err(FrameSeekError::from)?;
        let plan =
            BoundFrameIndex::new(inventory, index, self.index_limits)?.seek(target, limits)?;
        open_seek_plan(
            self.input.engine.as_ref(),
            codestream,
            self.input.request,
            plan,
        )
    }
}

impl<E: GpuSubmissionEngine + std::fmt::Debug> std::fmt::Debug for GpuDecodeSeekStream<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GpuDecodeSeekStream")
            .field("input", &self.input)
            .field("index", &self.index_stats())
            .finish_non_exhaustive()
    }
}
