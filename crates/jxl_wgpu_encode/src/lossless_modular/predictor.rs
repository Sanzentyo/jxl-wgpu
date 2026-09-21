use crate::EncodeError;

/// JPEG XL's fourteen Modular predictors. Prediction operates on integer working words,
/// including the raw words of floating-point input, after the selected RCT.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum LosslessModularPredictor {
    Zero = 0,
    West = 1,
    North = 2,
    AverageWestNorth = 3,
    Select = 4,
    #[default]
    Gradient = 5,
    Weighted = 6,
    NorthEast = 7,
    NorthWest = 8,
    WestWest = 9,
    AverageWestNorthWest = 10,
    AverageNorthNorthWest = 11,
    AverageNorthNorthEast = 12,
    AverageAll = 13,
}

impl LosslessModularPredictor {
    pub const ALL: [Self; 14] = [
        Self::Zero,
        Self::West,
        Self::North,
        Self::AverageWestNorth,
        Self::Select,
        Self::Gradient,
        Self::Weighted,
        Self::NorthEast,
        Self::NorthWest,
        Self::WestWest,
        Self::AverageWestNorthWest,
        Self::AverageNorthNorthWest,
        Self::AverageNorthNorthEast,
        Self::AverageAll,
    ];

    #[must_use]
    pub const fn value(self) -> u32 {
        self as u32
    }
}

/// Checked coefficients of JPEG XL's Weighted/SelfCorrecting predictor header.
/// All seven five-bit coefficients and four four-bit maximum weights are accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularWeightedPredictor {
    coefficients: [u8; 7],
    max_weights: [u8; 4],
}

impl Default for LosslessModularWeightedPredictor {
    fn default() -> Self {
        Self {
            coefficients: [16, 10, 7, 7, 7, 0, 0],
            max_weights: [13, 12, 12, 12],
        }
    }
}

impl LosslessModularWeightedPredictor {
    pub fn new(coefficients: [u8; 7], max_weights: [u8; 4]) -> Result<Self, EncodeError> {
        for (name, values, maximum) in [
            ("coefficient", coefficients.as_slice(), 31),
            ("max_weight", max_weights.as_slice(), 15),
        ] {
            for (index, &value) in values.iter().enumerate() {
                if value > maximum {
                    return Err(EncodeError::WeightedPredictorParameter {
                        name,
                        index,
                        value,
                        maximum,
                    });
                }
            }
        }
        Ok(Self {
            coefficients,
            max_weights,
        })
    }

    #[must_use]
    pub const fn coefficients(self) -> [u8; 7] {
        self.coefficients
    }

    #[must_use]
    pub const fn max_weights(self) -> [u8; 4] {
        self.max_weights
    }
}
