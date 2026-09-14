//! Metadata lowering for the shared GPU luminance mapper.
use crate::{Error, Result};
use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::ToneMapping;

/// Shared WGSL luminance mapping. Include [`crate::IMAGE_TRANSFER_SHADER`] before this fragment.
pub const TONE_MAPPING_SHADER: &str = include_str!("../shaders/tone_mapping.wgsl");

/// The same 48-byte record is used by image output and resident ICC programs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ToneMappingParams {
    pub(crate) range: [f32; 4],
    pub(crate) curve: [f32; 4],
    pub(crate) knee: [f32; 4],
}

impl ToneMappingParams {
    /// Calculate only metadata coefficients. Every pixel is evaluated by WGSL.
    pub fn new(mapping: ToneMapping) -> Result<Self> {
        let source = mapping.source();
        let target = mapping.target();
        let source_white = f64::from(source.white().nits());
        let target_white = f64::from(target.white().nits());
        let threshold = mapping.linear_below_nits();
        // Compare unit luminance directly. Multiplying a preceding F32 value by image
        // white can round it onto the protected boundary. The exclusive upper F32 neighbor
        // of the exact quotient preserves the strict comparison for every F32 luminance.
        let relative_threshold = (threshold / source_white).min(f64::from(f32::MAX));
        let rounded = relative_threshold as f32;
        let relative_threshold = if f64::from(rounded) < relative_threshold {
            rounded.next_up()
        } else {
            rounded
        };
        let mut params = Self {
            range: [
                source_white as f32,
                target_white as f32,
                relative_threshold,
                1.0,
            ],
            curve: [0.0; 4],
            knee: [0.0, 0.0, (source_white / target_white) as f32, 0.0],
        };
        let mut source_black = f64::from(source.black_nits());
        let mut target_black = f64::from(target.black_nits());
        if threshold >= source_white || threshold >= target_white {
            // Preserving absolute light takes precedence when the protected region reaches
            // either peak. This avoids a decreasing shoulder or a discontinuity at the threshold.
        } else if source_white <= target_white && source_black == target_black {
            // The source range already fits; preserve its absolute light, including equal
            // degenerate ranges, where an identity request must retain the input values.
        } else if target_black == target_white {
            params.range[3] = 3.0;
        } else if source_black == source_white {
            params.range[3] = 4.0;
        } else {
            let protected = threshold > 0.0;
            if protected {
                // Keep the protected shadow interval unchanged and connect the remaining
                // highlight range at the same absolute luminance on both sides.
                source_black = threshold;
                target_black = threshold;
            }
            let minimum = pq(source_black);
            let span = pq(source_white) - minimum;
            let low = (pq(target_black) - minimum) / span;
            let high = (pq(target_white) - minimum) / span;
            let knee = 1.5 * high - 0.5;
            let knee = if protected { knee.max(0.0) } else { knee };
            params.range[3] = 2.0;
            params.curve = [minimum, span, low, high].map(|v| v as f32);
            params.knee[0] = knee as f32;
            params.knee[1] = (1.0 / (1.0 - knee).max(1e-6)) as f32;
            params.knee[3] = (1.0 / span) as f32;
        }
        if bytemuck::cast_slice::<_, f32>(std::slice::from_ref(&params))
            .iter()
            .any(|v| !v.is_finite())
        {
            return Err(Error::InvalidPayload(
                "tone-mapping metadata exceeds GPU F32 precision".into(),
            ));
        }
        Ok(params)
    }
}

// ST 2084 is used here only for four metadata endpoints, never for image samples.
fn pq(nits: f64) -> f64 {
    if nits == 0.0 {
        return 0.0;
    }
    let power = (nits / 10000.0).powf(2610.0 / 16384.0);
    ((3424.0 / 4096.0 + 2413.0 / 128.0 * power) / (1.0 + 2392.0 / 128.0 * power))
        .powf(2523.0 / 32.0)
}

const _: () = {
    assert!(std::mem::size_of::<ToneMappingParams>() == 48);
    assert!(std::mem::offset_of!(ToneMappingParams, curve) == 16);
    assert!(std::mem::offset_of!(ToneMappingParams, knee) == 32);
};
