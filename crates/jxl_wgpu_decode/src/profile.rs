use jxl_gpu_bitstream::{BitReader, Gray8AccelerationIndex};

use crate::{Result, UnsupportedCodestreamFeature, UnsupportedProfile};

const MAX_DIMENSION: u32 = 256;

/// Verifies the fixed JPEG XL envelope around the indexed entropy stream.
///
/// The private index deliberately avoids duplicating a generic JPEG XL parser, but it is not
/// allowed to override the standard codestream. This check proves that the bound bytes use the
/// exact still-image, single-group, grayscale Modular envelope implemented by the GPU kernel and
/// that the indexed token range lies inside its sole group packet.
pub(crate) fn validate_gray8_envelope(
    codestream: &[u8],
    index: &Gray8AccelerationIndex,
) -> Result<()> {
    let mut reader = BitReader::new(codestream);
    expect(&mut reader, 16, 0x0aff, "JPEG XL codestream signature")?;
    expect(&mut reader, 1, 0, "non-small image header")?;
    let height = read_size(&mut reader, true)?;
    let width = read_size(&mut reader, false)?;
    if width != index.width() || height != index.height() {
        return unsupported("jwgp extent does not match the standard JPEG XL image header");
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return unsupported("the Gray8 GPU profile is limited to one 256x256 group");
    }

    // Image metadata: non-default 8-bit grayscale sRGB, no extra channels/ICC/extensions.
    for (count, value, name) in [
        (1, 0, "non-default image metadata"),
        (1, 0, "no extra metadata fields"),
        (1, 0, "integer bit depth"),
        (2, 0, "8-bit depth selector"),
        (1, 1, "16-bit modular buffer suffices"),
        (2, 0, "no extra channels"),
        (1, 0, "not XYB"),
        (1, 0, "non-default color encoding"),
        (1, 0, "no ICC profile"),
        (2, 1, "grayscale color space"),
        (2, 1, "D65 white point"),
        (1, 0, "enumerated transfer function"),
        (2, 0b10, "transfer-function selector"),
        (4, 11, "sRGB transfer function"),
        (2, 1, "relative rendering intent"),
        (2, 0, "no image extensions"),
        (1, 1, "default transform data"),
    ] {
        expect(&mut reader, count, value, name)?;
    }
    reader.align_to_byte()?;

    // Fixed final regular Modular frame, one pass/group, replace blend, no restoration filters.
    for (count, value, name) in [
        (1, 0, "non-default frame header"),
        (2, 0, "regular frame"),
        (1, 1, "Modular encoding"),
        (2, 0, "default frame flags"),
        (1, 0, "not YCbCr"),
        (2, 0, "no upsampling"),
        (2, 1, "default 256-pixel group size"),
        (2, 0, "one pass"),
        (1, 0, "default frame size and origin"),
        (2, 0, "replace blending"),
        (1, 1, "final frame"),
        (2, 0, "empty frame name"),
        (1, 0, "non-default loop filter"),
        (1, 0, "no Gaborish"),
        (2, 0, "no EPF"),
        (2, 0, "no loop-filter extensions"),
        (2, 0, "no frame-header extensions"),
        (1, 0, "canonical TOC order"),
    ] {
        expect(&mut reader, count, value, name)?;
    }
    reader.align_to_byte()?;

    let group_size = read_toc_size(&mut reader)?;
    reader.align_to_byte()?;
    let group_start = u64::try_from(reader.bit_offset()).map_err(|_| {
        UnsupportedProfile::new(
            UnsupportedCodestreamFeature::Other("size-overflow".into()),
            "group bit offset does not fit the GPU profile",
        )
    })?;
    // Regenerate the fixed DC-global/context-map/four-prefix-tree prefix from the bound index,
    // compare it bit-for-bit with the standard codestream, and prove that it ends at the exact
    // indexed token offset. The private box can therefore accelerate parsing but cannot redefine
    // the meaning of `jxlc` bits.
    index.validate_group_prefix(codestream, group_start)?;
    let group_bits = u64::from(group_size)
        .checked_mul(8)
        .ok_or_else(|| unsupported_error("group packet size overflow"))?;
    let group_end = group_start
        .checked_add(group_bits)
        .ok_or_else(|| unsupported_error("group packet range overflow"))?;
    let codestream_bits = u64::try_from(codestream.len())
        .ok()
        .and_then(|bytes| bytes.checked_mul(8))
        .ok_or_else(|| unsupported_error("codestream size overflow"))?;
    if group_end != codestream_bits {
        return unsupported("the Gray8 profile requires exactly one final group packet");
    }
    let token_end = index
        .token_bit_offset()
        .checked_add(index.token_bit_len())
        .ok_or_else(|| unsupported_error("indexed token range overflow"))?;
    if index.token_bit_offset() < group_start || token_end > group_end {
        return unsupported("jwgp token range is outside the sole JPEG XL group packet");
    }
    if !bits_are_zero(codestream, token_end, group_end) {
        return unsupported("non-zero data follows the indexed entropy stream in the group packet");
    }
    Ok(())
}

fn read_size(reader: &mut BitReader<'_>, has_ratio: bool) -> Result<u32> {
    let selector = reader.read_bits(2)? as usize;
    let widths = [9, 13, 18, 30];
    let value = u32::try_from(reader.read_bits(widths[selector])?)
        .map_err(|_| unsupported_error("image extent overflows u32"))?
        .checked_add(1)
        .ok_or_else(|| unsupported_error("image extent overflows u32"))?;
    if has_ratio {
        expect(reader, 3, 0, "explicit width follows height")?;
    }
    Ok(value)
}

fn read_toc_size(reader: &mut BitReader<'_>) -> Result<u32> {
    const BUCKETS: [(u32, u8); 4] = [(0, 10), (1_024, 14), (17_408, 22), (4_211_712, 30)];
    let selector = reader.read_bits(2)? as usize;
    let (base, bits) = BUCKETS[selector];
    let delta = u32::try_from(reader.read_bits(bits)?)
        .map_err(|_| unsupported_error("TOC size overflows u32"))?;
    base.checked_add(delta)
        .ok_or_else(|| unsupported_error("TOC size overflows u32").into())
}

fn expect(reader: &mut BitReader<'_>, count: u8, expected: u64, field: &str) -> Result<()> {
    let actual = reader.read_bits(count)?;
    if actual != expected {
        return unsupported(format!(
            "the Gray8 GPU profile requires {field} (expected {expected}, received {actual})"
        ));
    }
    Ok(())
}

fn bits_are_zero(bytes: &[u8], start: u64, end: u64) -> bool {
    (start..end).all(|bit| {
        usize::try_from(bit / 8)
            .ok()
            .and_then(|byte| bytes.get(byte))
            .is_some_and(|value| value & (1 << (bit % 8)) == 0)
    })
}

fn unsupported<T>(detail: impl Into<String>) -> Result<T> {
    Err(unsupported_error(detail).into())
}

fn unsupported_error(detail: impl Into<String>) -> UnsupportedProfile {
    UnsupportedProfile::new(
        UnsupportedCodestreamFeature::Other("gray8-envelope".into()),
        detail,
    )
}

#[cfg(test)]
mod tests {
    use super::bits_are_zero;

    #[test]
    fn checks_unaligned_padding_bits() {
        assert!(bits_are_zero(&[0b0000_0101], 3, 8));
        assert!(!bits_are_zero(&[0b0001_0101], 3, 8));
    }
}
