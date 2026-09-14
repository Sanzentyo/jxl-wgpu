//! Shared strategy constants and bounded metadata expansion for both codecs.
//! No image samples or coefficients enter this module.

mod quantization;
#[cfg(test)]
mod tests;

use crate::TransformKind;

impl TransformKind {
    /// Default coefficient permutation in canonical JPEG XL transform-buffer
    /// order. The initial `lf_extent().area()` positions are the LLF rectangle.
    #[must_use]
    pub fn natural_order(self) -> Vec<u32> {
        let extent = self.pixel_extent();
        let width = extent.width.max(extent.height);
        let height = extent.width.min(extent.height);
        let low_width = width / 8;
        let low_height = height / 8;
        let ratio = width / height;
        let mut order = Vec::with_capacity((width * height) as usize);
        for y in 0..low_height {
            for x in 0..low_width {
                order.push(y * width + x);
            }
        }
        for diagonal in 0..2 * width - 1 {
            let minimum = diagonal.saturating_sub(width - 1);
            let maximum = diagonal.min(width - 1);
            for step in minimum..=maximum {
                let (x, y) = if diagonal.is_multiple_of(2) {
                    (step, diagonal - step)
                } else {
                    (diagonal - step, step)
                };
                if !y.is_multiple_of(ratio) {
                    continue;
                }
                let y = y / ratio;
                if x >= low_width || y >= low_height {
                    order.push(y * width + x);
                }
            }
        }
        order
    }
}
