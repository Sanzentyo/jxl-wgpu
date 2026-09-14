use jxl_gpu_bitstream::BitWriter;

use super::super::dispatch::validate_artifact;
use super::super::entropy::{HfEntropyPlan, fixed_prefix_code};
use super::super::types::{
    ARTIFACT_READY, ArtifactLayout, VarDctArtifactHeader, VarDctFrameLayout,
};
use super::ac::write_tokens;

#[test]
fn artifact_rejects_missing_ac_writes_and_forged_layout() {
    let frame = VarDctFrameLayout::tiled_dct8(2057, 17).unwrap();
    let dc = fixed_prefix_code().unwrap();
    let hf = HfEntropyPlan::single_cluster_prefix().unwrap();
    let layout = ArtifactLayout::for_tiled_grid(frame, &dc, &hf).unwrap();
    let mut words = vec![0u32; layout.artifact_words as usize];
    let mut dc_fragment = BitWriter::new();
    for group in 0..frame.lf_group_count().unwrap() {
        let count = 3 * frame.lf_group_blocks(group).unwrap().block_count().unwrap();
        let start = dc_fragment.bit_len() as u32;
        for _ in 0..count {
            dc.write_raw(&mut dc_fragment, 0, 0, 0).unwrap();
        }
        let index = (layout.fragment_descriptor_offset + group * 2) as usize;
        words[index..index + 2].copy_from_slice(&[start, dc_fragment.bit_len() as u32 - start]);
    }
    let bit_len = dc_fragment.bit_len() as u32;
    for (index, byte) in dc_fragment.into_bytes().into_iter().enumerate() {
        words[layout.fragment_offset as usize + index / 4] |= u32::from(byte) << (index % 4 * 8);
    }
    let (ac_words, ac_bits) = write_tokens([0, 0, 0], &hf);
    for block in 0..layout.strategy_len {
        words[(layout.strategy_offset + block) as usize] = 0x100;
        words[(layout.ac_descriptor_offset + block) as usize] = ac_bits;
        let start = (layout.ac_fragment_offset + block * layout.ac_words_per_block) as usize;
        words[start..start + ac_words.len()].copy_from_slice(&ac_words);
    }
    let mut histogram = [0; 19];
    histogram[0] = layout.dc_len;
    let header = VarDctArtifactHeader {
        status: ARTIFACT_READY,
        block_count: layout.strategy_len,
        dc_sample_count: layout.dc_len,
        strategy: 0,
        ac_payload: 1,
        strategy_offset: layout.strategy_offset,
        strategy_len: layout.strategy_len,
        dc_offset: layout.dc_offset,
        dc_len: layout.dc_len,
        token_offset: layout.token_offset,
        token_len: layout.token_len,
        extra_offset: layout.extra_offset,
        extra_len: layout.extra_len,
        fragment_offset: layout.fragment_offset,
        fragment_word_capacity: layout.fragment_word_capacity,
        dc_fragment_bit_len: bit_len,
        artifact_words: layout.artifact_words,
        width: frame.width,
        height: frame.height,
        blocks_x: frame.blocks_x,
        blocks_y: frame.blocks_y,
        topology: frame.topology.artifact_id(),
        raw_histogram: histogram,
        fragment_descriptor_offset: layout.fragment_descriptor_offset,
        fragment_descriptor_len: layout.fragment_descriptor_len,
        lf_groups_x: frame.lf_groups_x,
        lf_groups_y: frame.lf_groups_y,
        lf_group_count: frame.lf_group_count().unwrap(),
        ac_descriptor_offset: layout.ac_descriptor_offset,
        ac_descriptor_len: layout.ac_descriptor_len,
        ac_fragment_offset: layout.ac_fragment_offset,
        ac_words_per_block: layout.ac_words_per_block,
        ac_fragment_words: layout.ac_fragment_words,
        padding: [0; 13],
    };
    words[..64].copy_from_slice(bytemuck::cast_slice(std::slice::from_ref(&header)));
    let valid = |words: &[u32]| {
        validate_artifact(bytemuck::cast_slice(words), layout, &dc, &hf, frame).is_ok()
    };
    assert!(valid(&words));
    // Every new AC header field, its presence marker and final ready status.
    for index in [0, 4, 46, 47, 48, 49, 50, 51] {
        let mut invalid = words.clone();
        invalid[index] ^= 1;
        assert!(!valid(&invalid), "header word {index}");
    }
    for block in [0, layout.strategy_len - 1] {
        let mut missing = words.clone();
        missing[(layout.ac_descriptor_offset + block) as usize] = 0;
        assert!(!valid(&missing), "missing block {block}");
        let mut padding = words.clone();
        let end = layout.ac_fragment_offset + (block + 1) * layout.ac_words_per_block;
        padding[end as usize - 1] = 1;
        assert!(!valid(&padding), "block padding {block}");
    }
    assert!(!valid(&words[..words.len() - 1]));
}
