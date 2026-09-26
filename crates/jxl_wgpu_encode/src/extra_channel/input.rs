//! Checked scalar input bindings shared by both frame encoders.
use crate::extra_channel::sampling::ExtraChannelSamplingPlan;
use crate::sample_format::ImageSamplePlan;
use crate::source::{SourceChannels, SourceLayout, SourceWindows};
use crate::{BufferImageSource, EncodeError, UnsupportedFeature};

pub(crate) struct ExtraInputPlan {
    pub(crate) independent: Vec<SourceLayout>,
    pub(crate) source_bytes: u64,
}

impl ExtraInputPlan {
    pub(crate) fn new(
        samples: &ImageSamplePlan,
        sampling: &ExtraChannelSamplingPlan,
        source: &BufferImageSource,
        main: &SourceLayout,
        alignment: u64,
    ) -> Result<Self, EncodeError> {
        let packed = usize::from(samples.alpha.is_some());
        if source.extra_channels().len() + packed != samples.extra_channels.len()
            || sampling.channels.len() != samples.extra_channels.len()
        {
            return Err(EncodeError::InvalidSource(
                "extra source count differs from the image declaration",
            ));
        }
        let mut independent = Vec::with_capacity(source.extra_channels().len());
        for (definition, sampled) in samples.extra_channels.iter().zip(&*sampling.channels) {
            let Some(input) = sampled.source else {
                continue;
            };
            let scalar = &source.extra_channels()[input];
            if scalar.layout.extent != sampled.extent
                || !scalar.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                || !scalar.extra_channels().is_empty()
            {
                return Err(EncodeError::InvalidSource(
                    "extra source extent or buffer usage differs from its declaration",
                ));
            }
            let layout = SourceLayout::new(&scalar.layout, scalar.buffer.size(), alignment)?;
            let precision = definition.precision().color(crate::ColorChannels::Gray);
            if layout.spec.format != SourceChannels::Gray
                || layout.spec.bits_per_sample != precision.bits_per_sample()
                || layout.spec.exponent_bits_per_sample != precision.exponent_bits()
            {
                return Err(UnsupportedFeature::InputFormat.into());
            }
            independent.push(layout);
        }
        let source_bytes = SourceWindows::addressed_bytes_many(
            std::iter::once((source.buffer.as_ref(), main.full_windows)).chain(
                independent
                    .iter()
                    .zip(source.extra_channels())
                    .map(|(layout, input)| (input.buffer.as_ref(), layout.full_windows)),
            ),
        )?;
        Ok(Self {
            independent,
            source_bytes,
        })
    }
}

/// Global Modular consumes a prefix, then remaining scalar grids use LF or pass groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScalarRoute {
    Global,
    Lf,
    Pass,
}

pub(crate) fn route(
    global_prefix: &mut bool,
    extent: jxl_gpu_protocol::Extent2d,
    shift: u8,
    group_dimension: u32,
) -> (ScalarRoute, u32) {
    *global_prefix &= extent.width <= group_dimension && extent.height <= group_dimension;
    if *global_prefix {
        (ScalarRoute::Global, group_dimension)
    } else if shift >= 3 {
        (ScalarRoute::Lf, (group_dimension * 8) >> shift)
    } else {
        (ScalarRoute::Pass, group_dimension >> shift)
    }
}
