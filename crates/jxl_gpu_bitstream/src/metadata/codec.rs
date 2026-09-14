use std::io::{self, Write};

use super::{BrotliOptions, MetadataError, MetadataLimits, MetadataResource, append, check};

pub(super) fn decompress(input: &[u8], limits: MetadataLimits) -> Result<Vec<u8>, MetadataError> {
    check_window(input, limits)?;
    let mut state = brotli::BrotliState::new_strict(
        brotli::HeapAlloc::<u8>::default(),
        brotli::HeapAlloc::<u32>::default(),
        brotli::HeapAlloc::<brotli::HuffmanCode>::default(),
    );
    let mut available_in = input.len();
    let mut input_offset = 0;
    let mut total_out = 0;
    let mut output = Vec::new();
    let mut buffer = [0_u8; 4096];
    let ratio_limit = (input.len() as u64).saturating_mul(u64::from(limits.max_expansion_ratio));
    let output_limit = limits.max_decoded_box_bytes.min(ratio_limit);
    loop {
        // At most one excess byte is decoded to distinguish exact-limit success from overflow.
        let capacity = output_limit
            .saturating_sub(output.len() as u64)
            .saturating_add(1)
            .min(buffer.len() as u64) as usize;
        let mut available_out = capacity;
        let mut output_offset = 0;
        let previous_input = input_offset;
        let result = brotli::BrotliDecompressStream(
            &mut available_in,
            &mut input_offset,
            input,
            &mut available_out,
            &mut output_offset,
            &mut buffer[..capacity],
            &mut total_out,
            &mut state,
        );
        let produced = (output.len() as u64)
            .checked_add(output_offset as u64)
            .ok_or(MetadataError::SizeOverflow)?;
        check(
            MetadataResource::DecodedBoxBytes,
            produced,
            limits.max_decoded_box_bytes,
        )?;
        check_expansion(produced, input.len() as u64, limits)?;
        append(
            &mut output,
            &buffer[..output_offset],
            limits.max_decoded_box_bytes,
            MetadataResource::DecodedBoxBytes,
        )?;
        match result {
            brotli::BrotliResult::ResultSuccess => {
                if available_in != 0 {
                    return Err(MetadataError::TrailingBrotliData);
                }
                return Ok(output);
            }
            brotli::BrotliResult::ResultFailure => return Err(MetadataError::InvalidBrotli),
            brotli::BrotliResult::NeedsMoreInput => return Err(MetadataError::TruncatedBrotli),
            brotli::BrotliResult::NeedsMoreOutput => {
                if previous_input == input_offset && output_offset == 0 {
                    return Err(MetadataError::InvalidBrotli);
                }
            }
        }
    }
}

pub(super) fn compress(
    box_type: [u8; 4],
    input: &[u8],
    options: BrotliOptions,
    limits: MetadataLimits,
) -> Result<Vec<u8>, MetadataError> {
    check(
        MetadataResource::BrotliWindowBits,
        u64::from(options.window_bits),
        u64::from(limits.max_brotli_window_bits),
    )?;
    let params = brotli::enc::BrotliEncoderParams {
        quality: i32::from(options.quality),
        lgwin: i32::from(options.window_bits),
        size_hint: input.len(),
        ..Default::default()
    };
    let mut writer = BoundedWriter {
        bytes: Vec::new(),
        limit: limits.max_encoded_box_bytes,
        error: None,
    };
    append(
        &mut writer.bytes,
        &box_type,
        limits.max_encoded_box_bytes,
        MetadataResource::EncodedBoxBytes,
    )?;
    if brotli::BrotliCompress(&mut &input[..], &mut writer, &params).is_err() {
        return Err(writer.error.unwrap_or(MetadataError::CompressionFailed));
    }
    check_window(&writer.bytes[4..], limits)?;
    check_expansion(input.len() as u64, (writer.bytes.len() - 4) as u64, limits)?;
    Ok(writer.bytes)
}

fn check_expansion(
    decoded: u64,
    compressed: u64,
    limits: MetadataLimits,
) -> Result<(), MetadataError> {
    if u128::from(decoded) > u128::from(compressed) * u128::from(limits.max_expansion_ratio) {
        Err(MetadataError::ExpansionLimit {
            compressed,
            decoded,
            max_ratio: limits.max_expansion_ratio,
        })
    } else {
        Ok(())
    }
}

fn check_window(input: &[u8], limits: MetadataLimits) -> Result<(), MetadataError> {
    let byte = *input.first().ok_or(MetadataError::TruncatedBrotli)?;
    // RFC 7932 section 9.1: the complete WBITS prefix fits in the first seven bits.
    let bits = if byte & 1 == 0 {
        16
    } else if (byte >> 1) & 7 != 0 {
        17 + ((byte >> 1) & 7)
    } else {
        match (byte >> 4) & 7 {
            0 => 17,
            1 => return Err(MetadataError::InvalidBrotli),
            n => 8 + n,
        }
    };
    check(
        MetadataResource::BrotliWindowBits,
        u64::from(bits),
        u64::from(limits.max_brotli_window_bits),
    )
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: u64,
    error: Option<MetadataError>,
}

impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if let Err(error) = append(
            &mut self.bytes,
            bytes,
            self.limit,
            MetadataResource::EncodedBoxBytes,
        ) {
            self.error = Some(error);
            return Err(io::Error::other(
                "metadata output limit or allocation failure",
            ));
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
