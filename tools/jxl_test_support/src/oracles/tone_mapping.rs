//! Independent F64 luminance reference: ITU-R BT.2408-8 Annex 5, with the JPEG XL E.3
//! protected interval. No production coefficients or GPU results enter these calculations.

#[derive(Clone, Copy, Debug)]
pub struct Mapping {
    pub source: [f64; 2],
    pub target: [f64; 2],
    pub protected: f64,
}

fn pq(nits: f64) -> f64 {
    if nits == 0.0 {
        return 0.0;
    }
    let powered = (nits.abs() / 10000.0).powf(2610.0 / 16384.0);
    ((3424.0 / 4096.0 + (2413.0 / 128.0) * powered) / (1.0 + (2392.0 / 128.0) * powered))
        .powf(2523.0 / 32.0)
        .copysign(nits)
}

fn nits(encoded: f64) -> f64 {
    let powered = encoded.max(0.0).powf(32.0 / 2523.0);
    10000.0
        * ((powered - 3424.0 / 4096.0).max(0.0) / (2413.0 / 128.0 - (2392.0 / 128.0) * powered))
            .powf(16384.0 / 2610.0)
}

impl Mapping {
    fn linear(self) -> bool {
        self.protected >= self.source[1]
            || self.protected >= self.target[1]
            || (self.source[1] <= self.target[1] && self.source[0] == self.target[0])
    }

    fn endpoints(self) -> [f64; 4] {
        let [source_black, target_black] = if self.protected > 0.0 {
            [self.protected; 2]
        } else {
            [self.source[0], self.target[0]]
        };
        let low = pq(source_black);
        let span = pq(self.source[1]) - low;
        [
            low,
            span,
            (pq(target_black) - low) / span,
            (pq(self.target[1]) - low) / span,
        ]
    }

    fn shoulder(self, luminance: f64) -> f64 {
        let [low, span, _, high] = self.endpoints();
        let x = ((pq(luminance) - low) / span).min(1.0);
        let start = 1.5 * high - 0.5;
        let start = if self.protected > 0.0 {
            start.max(0.0)
        } else {
            start
        };
        if x < start {
            return x;
        }
        let width = (1.0 - start).max(1e-6);
        let t = (x - start) / width;
        let derivative = (3.0 * (high - start) / width).clamp(0.0, 1.0);
        // Bernstein form of the cubic Hermite segment, independent of the WGSL polynomial.
        let controls = [start, start + (1.0 - start) * derivative / 3.0, high, high];
        (1.0 - t).powi(3) * controls[0]
            + 3.0 * t * (1.0 - t).powi(2) * controls[1]
            + 3.0 * t * t * (1.0 - t) * controls[2]
            + t.powi(3) * controls[3]
    }

    fn map_shoulder(self, value: f64) -> f64 {
        let [low, span, black, _] = self.endpoints();
        let encoded =
            ((value + black * (1.0 - value).powi(4)) * span + low).clamp(0.0, pq(self.target[1]));
        nits(encoded).clamp(0.0, self.target[1])
    }

    /// Output absolute luminance, retaining protected light even above the requested peak.
    pub fn luminance(self, value: f64) -> f64 {
        if self.linear() || (self.protected > 0.0 && value < self.protected) {
            return value;
        }
        if self.source[0] == self.source[1] || self.target[0] == self.target[1] {
            return self.target[1];
        }
        self.map_shoulder(self.shoulder(value))
    }

    pub fn apply(self, color: [f64; 3], weights: [f64; 3], neutral: [f64; 3]) -> [f64; 3] {
        let y = (0..3).map(|c| color[c] * weights[c]).sum::<f64>() * self.source[1];
        if self.linear() || (self.protected > 0.0 && y < self.protected) {
            return color.map(|v| v * self.source[1] / self.target[1]);
        }
        if self.target[0] == self.target[1] {
            return neutral;
        }
        if self.source[0] == self.source[1] {
            if y / self.source[1] <= 1e-6 {
                return neutral;
            }
            return color.map(|v| v * self.source[1] / y);
        }
        let output = self.luminance(y) / self.target[1];
        if y <= 1e-6 {
            return neutral.map(|v| v * output);
        }
        color.map(|v| v * output * self.source[1] / y)
    }

    fn luminance_interval(self, input: [f64; 2]) -> [f64; 2] {
        let mut result = [f64::INFINITY, f64::NEG_INFINITY];
        let mut include = |value: f64| {
            result[0] = result[0].min(value);
            result[1] = result[1].max(value);
        };
        for value in input {
            include(self.luminance(value));
        }
        if self.protected > 0.0 && input[0] < self.protected && self.protected <= input[1] {
            include(self.protected);
            include(self.luminance(self.protected));
        }
        if self.linear() || self.source[0] == self.source[1] || self.target[0] == self.target[1] {
            return result;
        }
        let lower = if self.protected > 0.0 {
            input[0].max(self.protected)
        } else {
            input[0]
        };
        if lower <= input[1] {
            let [a, b] = [self.shoulder(lower), self.shoulder(input[1])];
            let black = self.endpoints()[2];
            if black != 0.0 {
                // The shadow lift's derivative has one real zero; include it even for an
                // unusually high target black where that part of the ERF is not monotonic.
                let critical = 1.0 - (1.0 / (4.0 * black)).cbrt();
                if a <= critical && critical <= b {
                    include(self.map_shoulder(critical));
                }
            }
        }
        result
    }

    /// Conservative component interval. Luminance and RGB remain correlated in the actual image;
    /// interval products deliberately discard that correlation instead of fitting GPU errors.
    pub fn interval(
        self,
        input: [[f64; 2]; 3],
        weights: [f64; 3],
        neutral: [f64; 3],
    ) -> [[f64; 2]; 3] {
        let y: [f64; 2] = std::array::from_fn(|edge| {
            self.source[1]
                * (0..3)
                    .map(|c| weights[c] * input[c][if weights[c] >= 0.0 { edge } else { 1 - edge }])
                    .sum::<f64>()
        });
        if self.linear() || (self.protected > 0.0 && y[1] < self.protected) {
            return input.map(|range| range.map(|v| v * self.source[1] / self.target[1]));
        }
        assert!(
            self.source[0] != self.source[1] && self.target[0] != self.target[1],
            "degenerate fixtures use exact sample references"
        );
        let output = self.luminance_interval(y);
        let ratio = [output[0] / y[1].max(1e-6), output[1] / y[0].max(1e-6)]
            .map(|v| v * self.source[1] / self.target[1]);
        std::array::from_fn(|c| {
            let mut range = [f64::INFINITY, f64::NEG_INFINITY];
            let mut include = |value: f64| {
                range[0] = range[0].min(value);
                range[1] = range[1].max(value);
            };
            if y[1] > 1e-6 {
                for value in input[c] {
                    for factor in ratio {
                        include(value * factor);
                    }
                }
            }
            if y[0] <= 1e-6 {
                for value in output {
                    include(value * neutral[c] / self.target[1]);
                }
            }
            if self.protected > 0.0 && y[0] < self.protected {
                for value in input[c] {
                    include(value * self.source[1] / self.target[1]);
                }
            }
            range
        })
    }
}
