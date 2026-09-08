//! The fixed-size LF-global noise model; random samples are generated only by the GPU.

use jxl_gpu_bitstream::FrameInventory;
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{ResidentNoiseError, ResidentNoiseParameters, ResidentNoisePlan};

use crate::modular_tree::BitInput;

/// Eight unsigned 10-bit samples of the signaled luma-dependent noise strength.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoiseModel {
    values: [u16; 8],
}

impl NoiseModel {
    pub(crate) fn parse(reader: &mut impl BitInput, packet_end: u64) -> crate::Result<Self> {
        let mut bounded = crate::vardct_frontend::BoundedBitInput::new(reader, packet_end);
        let mut values = [0; 8];
        for value in &mut values {
            *value = bounded.read_bits(10)? as u16;
        }
        Ok(Self { values })
    }

    /// Returns exact binary32 strengths in [0, 1), in stream order.
    pub fn lut(self) -> [f32; 8] {
        self.values.map(|value| f32::from(value) / 1024.0)
    }

    pub(crate) fn parameters(
        self,
        frame: &FrameInventory,
        correlation: [f32; 2],
    ) -> Option<ResidentNoiseParameters> {
        self.values
            .iter()
            .any(|&value| value != 0)
            .then(|| ResidentNoiseParameters {
                lut: self.lut(),
                correlation,
                frame_seed: frame.noise_seed,
                group_dimension: 128_u32.checked_shl(frame.group_size_shift).unwrap_or(0),
            })
    }

    pub(crate) fn plan(
        self,
        frame: &FrameInventory,
        extent: Extent2d,
        correlation: [f32; 2],
        limits: &wgpu::Limits,
    ) -> Result<Option<ResidentNoisePlan>, ResidentNoiseError> {
        self.parameters(frame, correlation)
            .map(|parameters| ResidentNoisePlan::new(extent, parameters, limits))
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_model_reads_exactly_eighty_bits_and_never_crosses_its_packet() {
        let values = [0_u16, 1, 2, 7, 31, 511, 1022, 1023];
        let mut writer = jxl_gpu_bitstream::BitWriter::new();
        writer.write_bits(5, 3).unwrap();
        for value in values {
            writer.write_bits(u64::from(value), 10).unwrap();
        }
        writer.write_bits(255, 8).unwrap();
        let data = writer.into_bytes();
        for length in 0..80 {
            let mut reader = jxl_gpu_bitstream::BitReader::new(&data);
            reader.skip_bits(3).unwrap();
            assert!(matches!(
                NoiseModel::parse(&mut reader, 3 + length),
                Err(crate::Error::Bitstream(
                    jxl_gpu_bitstream::Error::UnexpectedEndOfBits
                ))
            ));
            assert!(reader.bit_offset() <= 3 + length);
        }
        let mut reader = jxl_gpu_bitstream::BitReader::new(&data);
        reader.skip_bits(3).unwrap();
        let model = NoiseModel::parse(&mut reader, 83).unwrap();
        assert_eq!(model.values, values);
        assert_eq!(model.lut(), values.map(|value| f32::from(value) / 1024.0));
        assert_eq!(reader.bit_offset(), 83);
        assert_eq!(reader.read_bits(8).unwrap(), 255);
    }
}
