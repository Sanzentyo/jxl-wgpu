//! Checked scalar input bindings shared by both frame encoders.
use crate::extra_channel::sampling::ExtraChannelSamplingPlan;
use crate::sample_format::ImageSamplePlan;
use crate::source::{SourceChannels, SourceLayout, SourceWindows};
use crate::{BufferImageSource, EncodeError, UnsupportedFeature};

pub(crate) struct ExtraInputPlan {
    pub(crate) independent: Vec<ScalarInput>,
    pub(crate) source_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ScalarBuffer {
    Primary,
    Attached(usize),
}

impl ScalarBuffer {
    pub(crate) fn buffer(self, source: &BufferImageSource) -> &wgpu::Buffer {
        match self {
            Self::Primary => &source.buffer,
            Self::Attached(index) => &source.extra_channels()[index].buffer,
        }
    }
}

pub(crate) struct ScalarInput {
    pub(crate) layout: SourceLayout,
    pub(crate) buffer: ScalarBuffer,
}

impl ExtraInputPlan {
    pub(crate) fn new(
        samples: &ImageSamplePlan,
        sampling: &ExtraChannelSamplingPlan,
        source: &BufferImageSource,
        main: &SourceLayout,
        alignment: u64,
    ) -> Result<Self, EncodeError> {
        source.validate_alpha_association(samples.alpha.unwrap_or_default())?;
        let packed = usize::from(samples.alpha.is_some());
        if source.extra_channels().len() + packed + usize::from(samples.cmyk)
            != samples.extra_channels.len()
            || sampling.channels.len() != samples.extra_channels.len()
        {
            return Err(EncodeError::InvalidSource(
                "extra source count differs from the image declaration",
            ));
        }
        let mut independent =
            Vec::with_capacity(source.extra_channels().len() + usize::from(samples.cmyk));
        for (definition, sampled) in samples.extra_channels.iter().zip(&*sampling.channels) {
            let Some(input) = sampled.source else {
                continue;
            };
            let (layout, buffer) = if samples.cmyk && input == 0 {
                if source.layout.extent != sampled.extent {
                    return Err(EncodeError::InvalidSource(
                        "primary Black must use the color sample grid",
                    ));
                }
                (
                    main.black
                        .as_deref()
                        .ok_or(EncodeError::InvalidSource("CMYK input has no Black source"))?
                        .clone(),
                    ScalarBuffer::Primary,
                )
            } else {
                let index = input - usize::from(samples.cmyk);
                let scalar = &source.extra_channels()[index];
                if scalar.layout.extent != sampled.extent
                    || !scalar.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                    || !scalar.extra_channels().is_empty()
                {
                    return Err(EncodeError::InvalidSource(
                        "extra source extent or buffer usage differs from its declaration",
                    ));
                }
                (
                    SourceLayout::for_source(scalar, alignment)?,
                    ScalarBuffer::Attached(index),
                )
            };
            let precision = definition.precision().color(crate::ColorChannels::Gray);
            if layout.spec.format != SourceChannels::Gray
                || layout.spec.bits_per_sample != precision.bits_per_sample()
                || layout.spec.exponent_bits_per_sample != precision.exponent_bits()
            {
                return Err(UnsupportedFeature::InputFormat.into());
            }
            independent.push(ScalarInput { layout, buffer });
        }
        let source_bytes = SourceWindows::addressed_bytes_many(
            std::iter::once((source.buffer.as_ref(), main.full_windows)).chain(
                independent
                    .iter()
                    .map(|input| (input.buffer.buffer(source), input.layout.full_windows)),
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
