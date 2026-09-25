//! Bounded JPEG XL ICC metadata coding. No image samples are read here.

use jxl_gpu_bitstream::BitWriter;
use jxl_gpu_protocol::icc::IccProfile;

use crate::{BitFragment, EncodeError};

pub(crate) const DEFAULT_PROFILE_LIMIT: u64 = 16 << 20;
const MAX_ICC_BYTES: u64 = 1 << 28;

pub(crate) struct PreparedImageHeader {
    output: BitWriter,
    profile: Option<IccProfile>,
    icc_plan: Option<IccStreamPlan>,
    serialized_bytes: usize,
    pub(crate) icc_profile_bytes: u64,
    pub(crate) icc_storage_bytes: u64,
    pub(crate) extra_storage_bytes: u64,
}

impl PreparedImageHeader {
    pub(crate) fn new(
        output: BitWriter,
        profile: Option<&IccProfile>,
        limit: u64,
    ) -> Result<Self, EncodeError> {
        let icc_profile_bytes = profile.map_or(0, |profile| profile.bytes().len() as u64);
        let icc_plan = profile
            .map(|_| IccStreamPlan::new(icc_profile_bytes, limit))
            .transpose()?;
        let serialized_bytes = output
            .bit_len()
            .checked_add(icc_plan.map_or(0, |plan| plan.bit_len))
            .and_then(|bits| bits.checked_add(7))
            .map(|bits| bits / 8)
            .ok_or(EncodeError::InvalidConfiguration(
                "ICC header size overflow",
            ))?;
        // The header allocation moves into final assembly. Reserve two copies while an exact
        // resize or jxlc wrapping temporarily retains both old and new metadata storage.
        let icc_storage_bytes =
            if profile.is_some() {
                (serialized_bytes as u64).checked_mul(2).ok_or(
                    EncodeError::InvalidConfiguration("ICC storage size overflow"),
                )?
            } else {
                0
            };
        Ok(Self {
            output,
            profile: profile.cloned(),
            icc_plan,
            serialized_bytes,
            icc_profile_bytes,
            icc_storage_bytes,
            extra_storage_bytes: 0,
        })
    }

    /// ICC already owns the entire header. Otherwise independently declared extras add
    /// variable metadata which must retain its own permit through final assembly.
    pub(crate) fn account_extra_metadata(mut self, required: bool) -> Result<Self, EncodeError> {
        if required && self.icc_storage_bytes == 0 {
            self.extra_storage_bytes = (self.serialized_bytes as u64).checked_mul(2).ok_or(
                EncodeError::InvalidConfiguration("extra metadata storage size overflow"),
            )?;
        }
        Ok(self)
    }

    pub(crate) fn finish(
        mut self,
        budget: &jxl_wgpu::MemoryBudget,
    ) -> Result<(BitFragment, Option<jxl_wgpu::MemoryPermit>), EncodeError> {
        let storage_bytes = self.icc_storage_bytes + self.extra_storage_bytes;
        let permit = (storage_bytes != 0)
            .then(|| budget.try_reserve(storage_bytes))
            .transpose()?;
        self.output
            .try_reserve_bytes(self.serialized_bytes - self.output.as_bytes().len())?;
        if let Some(plan) = self.icc_plan {
            plan.write(
                &mut self.output,
                self.profile.as_ref().expect("planned ICC"),
            )?;
        }
        self.output.align_to_byte()?;
        if self.output.as_bytes().len() != self.serialized_bytes {
            return Err(EncodeError::InvalidConfiguration(
                "ICC header allocation disagrees with plan",
            ));
        }
        Ok((BitFragment::byte_aligned(self.output.into_bytes())?, permit))
    }
}

#[derive(Clone, Copy)]
pub(crate) struct IccStreamPlan {
    profile_bytes: u64,
    transformed_bytes: u64,
    pub(crate) bit_len: usize,
}

impl IccStreamPlan {
    pub(crate) fn new(profile_bytes: u64, limit: u64) -> Result<Self, EncodeError> {
        for (resource, limit) in [("profile bytes", limit), ("profile bytes", MAX_ICC_BYTES)] {
            if profile_bytes > limit {
                return Err(EncodeError::IccLimit {
                    resource,
                    required: profile_bytes,
                    limit,
                });
            }
        }
        if profile_bytes < 132 {
            return Err(EncodeError::InvalidConfiguration(
                "ICC header is incomplete",
            ));
        }
        // Zero disables the transformed tag table. One literal command preserves the original
        // tag table and every payload byte after the mandatory 128-byte header predictor.
        let commands_bytes = 2 + varint_len(profile_bytes - 128);
        let transformed_bytes =
            profile_bytes + varint_len(profile_bytes) + varint_len(commands_bytes) + commands_bytes;
        if transformed_bytes > MAX_ICC_BYTES {
            return Err(EncodeError::IccLimit {
                resource: "transformed profile bytes",
                required: transformed_bytes,
                limit: MAX_ICC_BYTES,
            });
        }
        let mut header = BitWriter::new();
        write_entropy_header(&mut header, transformed_bytes)?;
        let bit_len = usize::try_from(transformed_bytes)
            .ok()
            .and_then(|bytes| bytes.checked_mul(8))
            .and_then(|bits| bits.checked_add(header.bit_len()))
            .ok_or(EncodeError::InvalidConfiguration("ICC bit length overflow"))?;
        Ok(Self {
            profile_bytes,
            transformed_bytes,
            bit_len,
        })
    }

    pub(crate) fn write(
        self,
        output: &mut BitWriter,
        profile: &IccProfile,
    ) -> Result<(), EncodeError> {
        if profile.bytes().len() as u64 != self.profile_bytes {
            return Err(EncodeError::InvalidConfiguration(
                "ICC size changed after planning",
            ));
        }
        let start = output.bit_len();
        write_entropy_header(output, self.transformed_bytes)?;
        // The uniform 256-symbol canonical code reverses each byte in the LSB-first writer.
        // All 41 ICC contexts share this table, so no variable-sized histogram is needed.
        let mut emit = |value: u8| output.write_bits(u64::from(value.reverse_bits()), 8);
        let commands_bytes = 2 + varint_len(self.profile_bytes - 128);
        write_varint(self.profile_bytes, &mut emit)?;
        write_varint(commands_bytes, &mut emit)?;
        emit(0)?; // no transformed tag table
        emit(1)?; // literal-copy command
        write_varint(self.profile_bytes - 128, &mut emit)?;
        for (index, &value) in profile.bytes().iter().enumerate() {
            let predicted = if index < 128 {
                header_prediction(index, profile.bytes())
            } else {
                0
            };
            emit(value.wrapping_sub(predicted))?;
        }
        if output.bit_len() - start != self.bit_len {
            return Err(EncodeError::InvalidConfiguration(
                "ICC coding plan disagrees with output",
            ));
        }
        Ok(())
    }
}

fn varint_len(mut value: u64) -> u64 {
    let mut bytes = 1;
    while value >= 128 {
        value >>= 7;
        bytes += 1;
    }
    bytes
}

fn write_varint(
    mut value: u64,
    emit: &mut impl FnMut(u8) -> Result<(), jxl_gpu_bitstream::Error>,
) -> Result<(), jxl_gpu_bitstream::Error> {
    while value >= 128 {
        emit((value as u8 & 127) | 128)?;
        value >>= 7;
    }
    emit(value as u8)
}

fn write_entropy_header(output: &mut BitWriter, bytes: u64) -> Result<(), EncodeError> {
    match bytes {
        0 => output.write_bits(0, 2)?,
        1..=16 => {
            output.write_bits(1, 2)?;
            output.write_bits(bytes - 1, 4)?;
        }
        17..=272 => {
            output.write_bits(2, 2)?;
            output.write_bits(bytes - 17, 8)?;
        }
        _ => {
            output.write_bits(3, 2)?;
            output.write_bits(bytes & 4095, 12)?;
            let mut remaining = bytes >> 12;
            while remaining != 0 {
                output.write_bits(1, 1)?;
                output.write_bits(remaining & 255, 8)?;
                remaining >>= 8;
            }
            output.write_bits(0, 1)?;
        }
    }
    output.write_bits(0, 1)?; // no LZ77
    output.write_bits(1, 1)?; // simple context map
    output.write_bits(0, 2)?; // every context selects cluster zero
    output.write_bits(1, 1)?; // prefix entropy
    output.write_bits(8, 4)?; // hybrid-uint split exponent
    output.write_bits(0, 4)?; // no extra token MSBs
    output.write_bits(0, 4)?; // no extra token LSBs
    output.write_bits(1, 1)?;
    output.write_bits(7, 4)?;
    output.write_bits(127, 7)?; // alphabet size = 1 + 128 + 127
    output.write_bits(0, 2)?; // complex prefix tree, no skipped code lengths
    // A single code-length symbol (8) assigns all 256 byte symbols eight bits. The
    // code-length alphabet is read in full, then its single symbol consumes no further bits.
    for symbol in [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
        if symbol == 8 {
            output.write_bits(0b0111, 4)?; // code-length alphabet entry has length one
        } else {
            output.write_bits(0, 2)?;
        }
    }
    Ok(())
}

fn header_prediction(index: usize, profile: &[u8]) -> u8 {
    match index {
        0..=3 => (profile.len() as u32).to_be_bytes()[index],
        8 => 4,
        12..=23 => b"mntrRGB XYZ "[index - 12],
        36..=39 => b"acsp"[index - 36],
        41 | 42 if profile[40] == b'A' => b'P',
        43 if profile[40] == b'A' => b'L',
        41 if profile[40] == b'M' => b'S',
        42 if profile[40] == b'M' => b'F',
        43 if profile[40] == b'M' => b'T',
        42 if &profile[40..42] == b"SG" => b'I',
        43 if &profile[40..42] == b"SG" => b' ',
        42 if &profile[40..42] == b"SU" => b'N',
        43 if &profile[40..42] == b"SU" => b'W',
        70 => 246,
        71 => 214,
        73 => 1,
        78 => 211,
        79 => 45,
        80..=83 => profile[index - 76],
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_entropy_round_trips_every_context_and_size_bucket() {
        for bytes in [
            0,
            1,
            16,
            17,
            272,
            273,
            4095,
            4096,
            65535,
            65536,
            MAX_ICC_BYTES,
        ] {
            for offset in 0..8 {
                let mut writer = BitWriter::new();
                writer.write_bits(0, offset).unwrap();
                write_entropy_header(&mut writer, bytes).unwrap();
                for byte in 0..=u8::MAX {
                    writer
                        .write_bits(u64::from(byte.reverse_bits()), 8)
                        .unwrap();
                }
                let bit_len = writer.bit_len();
                let encoded = writer.into_bytes();
                let mut reader = jxl_bitstream::Bitstream::new(&encoded);
                reader.skip_bits(offset as usize).unwrap();
                assert_eq!(reader.read_u64().unwrap(), bytes);
                let mut decoder = jxl_coding::Decoder::parse(&mut reader, 41).unwrap();
                decoder.begin(&mut reader).unwrap();
                for byte in 0..=u8::MAX {
                    assert_eq!(
                        decoder
                            .read_varint(&mut reader, u32::from(byte) % 41)
                            .unwrap(),
                        u32::from(byte)
                    );
                }
                decoder.finalize().unwrap();
                assert_eq!(reader.num_read_bits(), bit_len);
            }
        }
    }

    #[test]
    fn profile_limits_are_checked_before_variable_sized_allocation() {
        assert!(IccStreamPlan::new(132, 132).is_ok());
        assert!(matches!(
            IccStreamPlan::new(132, 131),
            Err(EncodeError::IccLimit {
                resource: "profile bytes",
                required: 132,
                limit: 131
            })
        ));
        assert!(matches!(
            IccStreamPlan::new(131, u64::MAX),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            IccStreamPlan::new(u64::MAX, u64::MAX),
            Err(EncodeError::IccLimit {
                resource: "profile bytes",
                limit: MAX_ICC_BYTES,
                ..
            })
        ));
        assert!(matches!(
            IccStreamPlan::new(MAX_ICC_BYTES, u64::MAX),
            Err(EncodeError::IccLimit {
                resource: "transformed profile bytes",
                limit: MAX_ICC_BYTES,
                ..
            })
        ));
        assert!(IccStreamPlan::new(MAX_ICC_BYTES - 32, u64::MAX).is_ok());
    }
}
