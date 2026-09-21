//! Validated coefficient-order metadata. Image coefficients never enter this module.

use std::sync::Arc;

use jxl_gpu_bitstream::BitWriter;

use super::VarDctStrategy;
use super::bitstream::write_unsigned_token;
use super::entropy::{fixed_prefix_code, write_prefix_config};
use crate::EncodeError;

// JPEG XL size classes, including the separate special-8x8 class. Normative mapping:
// libjxl v0.12.0 lib/jxl/coeff_order.h, kStrategyOrder.
const FAMILIES: [usize; 27] = [
    0, 1, 1, 1, 2, 3, 4, 4, 5, 5, 6, 6, 1, 1, 1, 1, 1, 1, 7, 8, 8, 9, 10, 10, 11, 12, 12,
];

type ChannelOrders = [Box<[u32]>; 3];

/// Caller-selected permutations of natural coefficient ranks, independently for X/Y/B.
///
/// Orders are shared by JPEG XL size class: transposed rectangular strategies share an order,
/// and all special 8×8 strategies share a class distinct from regular DCT8. Unspecified classes
/// use their natural order. This is explicit metadata, not content-adaptive order selection.
///
/// ```
/// use jxl_wgpu_encode::{VarDctCoefficientOrders, VarDctConfig, VarDctStrategy};
/// let mut x: Vec<u32> = (0..64).collect();
/// x[1..].reverse(); // Keep the separately encoded LF coefficient first.
/// let config = VarDctConfig {
///     coefficient_orders: VarDctCoefficientOrders::default().with_order(
///         VarDctStrategy::Dct8, [x, (0..64).collect(), (0..64).collect()],
///     )?,
///     ..Default::default()
/// };
/// # Ok::<(), jxl_wgpu_encode::EncodeError>(())
/// ```
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct VarDctCoefficientOrders {
    families: [Option<Arc<ChannelOrders>>; 13],
}

impl std::fmt::Debug for VarDctCoefficientOrders {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VarDctCoefficientOrders")
            .field("custom_family_mask", &self.used_mask())
            .finish()
    }
}

impl VarDctCoefficientOrders {
    /// Replaces the selected strategy's complete size class with three permutations.
    ///
    /// Each array contains every natural rank in `0..width*height` exactly once. Its first
    /// `width*height/64` entries must remain the identity: LF coefficients are encoded separately.
    /// Explicit identity orders restore the default and retain the natural-order codestream bytes.
    /// Malformed lengths, duplicate/out-of-range ranks and moved LF ranks return a typed error.
    pub fn with_order(
        mut self,
        strategy: VarDctStrategy,
        orders: [Vec<u32>; 3],
    ) -> Result<Self, EncodeError> {
        let family = FAMILIES[strategy.codestream_id() as usize];
        let extent = strategy.pixel_extent();
        let len = (extent.width * extent.height) as usize;
        let mut identity = true;
        for (channel, order) in orders.iter().enumerate() {
            let invalid = |reason| EncodeError::VarDctCoefficientOrder {
                family: family as u8,
                channel: channel as u8,
                reason,
            };
            if order.len() != len {
                return Err(invalid("length differs from the transform area"));
            }
            let mut seen = vec![false; len];
            for (index, &rank) in order.iter().enumerate() {
                let rank = rank as usize;
                if rank >= len || seen[rank] {
                    return Err(invalid("ranks must be an in-range permutation"));
                }
                if index < len / 64 && rank != index {
                    return Err(invalid("the LF prefix must remain in natural order"));
                }
                seen[rank] = true;
                identity &= rank == index;
            }
        }
        self.families[family] = (!identity).then(|| Arc::new(orders.map(Vec::into_boxed_slice)));
        Ok(self)
    }

    /// Returns this strategy's X/Y/B natural-rank permutations, or `None` for natural order.
    #[must_use]
    pub fn permutations(&self, strategy: VarDctStrategy) -> Option<[&[u32]; 3]> {
        self.families[FAMILIES[strategy.codestream_id() as usize]]
            .as_ref()
            .map(|orders| std::array::from_fn(|channel| orders[channel].as_ref()))
    }

    pub(super) fn indices(&self, strategy: VarDctStrategy) -> Vec<[u32; 3]> {
        let natural = strategy.natural_order();
        let custom = self.permutations(strategy);
        (0..natural.len())
            .map(|rank| {
                std::array::from_fn(|channel| {
                    natural[custom.map_or(rank, |orders| orders[channel][rank] as usize)]
                })
            })
            .collect()
    }

    fn used_mask(&self) -> u16 {
        self.families
            .iter()
            .enumerate()
            .fold(0, |mask, (id, order)| {
                mask | (u16::from(order.is_some()) << id)
            })
    }

    pub(super) fn write(&self, output: &mut BitWriter) -> Result<(), EncodeError> {
        let mask = self.used_mask();
        match mask {
            0x5f => output.write_bits(0, 2)?,
            0x13 => output.write_bits(1, 2)?,
            0 => output.write_bits(2, 2)?,
            _ => {
                output.write_bits(3, 2)?;
                output.write_bits(u64::from(mask), 13)?;
            }
        }
        if mask == 0 {
            return Ok(());
        }
        // All eight permutation contexts share one stateless prefix distribution. These are
        // caller-supplied control permutations, never image-domain entropy jobs.
        let code = fixed_prefix_code()?;
        write_prefix_config(output, &code, 8)?;
        for orders in self.families.iter().flatten() {
            for order in orders.iter() {
                let lehmer = lehmer_tail(order);
                write_unsigned_token(output, &code, lehmer.len() as u32)?;
                for rank in lehmer {
                    write_unsigned_token(output, &code, rank)?;
                }
            }
        }
        Ok(())
    }
}

/// Rank among the remaining AC entries, using bounded O(N log N) metadata work.
fn lehmer_tail(order: &[u32]) -> Vec<u32> {
    let skip = order.len() / 64;
    let len = order.len() - skip;
    let mut counts = (0..=len)
        .map(|index| index.isolate_lowest_one() as u32)
        .collect::<Vec<_>>();
    let mut result = Vec::with_capacity(len);
    for &rank in &order[skip..] {
        let value = rank as usize - skip;
        let mut position = value;
        let mut lower = 0;
        while position != 0 {
            lower += counts[position];
            position &= position - 1;
        }
        result.push(lower);
        position = value + 1;
        while position <= len {
            counts[position] -= 1;
            position += position.isolate_lowest_one();
        }
    }
    while result.last() == Some(&0) {
        result.pop();
    }
    result
}
