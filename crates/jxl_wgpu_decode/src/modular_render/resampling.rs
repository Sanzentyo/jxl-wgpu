use jxl_gpu_protocol::Extent2d;

use super::{Result, invalid};

#[derive(Clone, Copy, Debug)]
pub(super) struct Step {
    pub factor: u32,
    pub extent: Extent2d,
}

#[derive(Debug)]
pub(super) struct Upsampling {
    pub steps: Vec<Step>,
}

impl Upsampling {
    pub fn new(source: Extent2d, output: Extent2d, factor: u32) -> Result<Self> {
        let steps = match factor {
            1 => Vec::new(),
            2 | 4 | 8 => vec![Step {
                factor,
                extent: output,
            }],
            16 | 32 | 64 => {
                let Some(width) = source.width.checked_mul(8) else {
                    return invalid("upsampling intermediate width overflow");
                };
                let Some(height) = source.height.checked_mul(8) else {
                    return invalid("upsampling intermediate height overflow");
                };
                // Retain the complete first-stage grid. Cropping it to ceil(output / remaining)
                // changes which samples the next filter mirrors at the right and bottom edges.
                vec![
                    Step {
                        factor: 8,
                        extent: Extent2d::new(width, height),
                    },
                    Step {
                        factor: factor / 8,
                        extent: output,
                    },
                ]
            }
            _ => return invalid("upsampling factor"),
        };
        Ok(Self { steps })
    }

    pub fn factor(&self) -> u32 {
        self.steps.iter().map(|step| step.factor).product()
    }

    pub fn intermediate_extent(&self) -> Option<Extent2d> {
        (self.steps.len() == 2).then(|| self.steps[0].extent)
    }
}
