//! Bounded host metadata entropy shared by coefficient orders and TOC permutations.
use jxl_gpu_bitstream::BitWriter;

use crate::EncodeError;
use crate::prefix::{RAW_SYMBOLS, RawPrefixCode};

pub(crate) fn write_config(
    output: &mut BitWriter,
) -> Result<RawPrefixCode<RAW_SYMBOLS>, EncodeError> {
    let code = RawPrefixCode::from_counts(&[1; RAW_SYMBOLS])?;
    code.write_stream_config(output, 8)?;
    Ok(code)
}

/// Caller validation must establish an in-range permutation and an unchanged skipped prefix.
pub(crate) fn write(
    output: &mut BitWriter,
    code: &RawPrefixCode<RAW_SYMBOLS>,
    order: &[u32],
    skip: usize,
) -> Result<(), EncodeError> {
    let lehmer = lehmer_tail(order, skip);
    code.write_unsigned(output, lehmer.len() as u32)?;
    for rank in lehmer {
        code.write_unsigned(output, rank)?;
    }
    Ok(())
}

/// Rank among the remaining entries, using bounded O(N log N) metadata work.
fn lehmer_tail(order: &[u32], skip: usize) -> Vec<u32> {
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
