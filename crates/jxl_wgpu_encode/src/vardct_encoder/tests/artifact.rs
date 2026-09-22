use jxl_gpu_bitstream::BitWriter;

use super::super::dispatch::validate_artifact;
use super::super::entropy::{HfEntropyPlan, fixed_prefix_code};
use super::super::types::{
    ARTIFACT_READY, ArtifactLayout, VarDctArtifactHeader, VarDctFrameLayout,
};
use super::ac::write_tokens;

#[test]
fn artifact_rejects_missing_ac_writes_and_forged_layout() {
    for (passes, saliency) in [(1, false), (3, false), (11, false), (1, true), (11, true)] {
        let frame = VarDctFrameLayout::tiled_dct8(2057, 17).unwrap();
        let dc = fixed_prefix_code().unwrap();
        let hf = HfEntropyPlan::single_cluster_prefix().unwrap();
        let mut layout = ArtifactLayout::for_tiled_grid(frame, &dc, &hf)
            .unwrap()
            .with_passes(passes)
            .unwrap();
        if saliency {
            layout = layout
                .with_saliency(frame.ac_group_count().unwrap())
                .unwrap();
        }
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
            words[layout.fragment_offset as usize + index / 4] |=
                u32::from(byte) << (index % 4 * 8);
        }
        let (ac_words, ac_bits) = write_tokens([0, 0, 0], &hf);
        for block in 0..layout.strategy_len {
            words[(layout.strategy_offset + block) as usize] = 0x100;
            for pass in 0..layout.ac_pass_count {
                let block = pass * layout.strategy_len + block;
                words[(layout.ac_descriptor_offset + block) as usize] = ac_bits;
                let start =
                    (layout.ac_fragment_offset + block * layout.ac_words_per_block) as usize;
                words[start..start + ac_words.len()].copy_from_slice(&ac_words);
            }
        }
        let mut histogram = [0; 33];
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
            ac_pass_count: layout.ac_pass_count,
            saliency_offset: layout.saliency_offset,
            saliency_groups: layout.saliency_groups,
        };
        words[..68].copy_from_slice(bytemuck::cast_slice(std::slice::from_ref(&header)));
        if saliency {
            let pixels = vec![[0; 3]; (frame.width * frame.height) as usize];
            for (group, (edges, contrast)) in
                super::saliency::oracle(frame.width as usize, frame.height as usize, &pixels)
                    .into_iter()
                    .enumerate()
            {
                let offset = layout.saliency_offset as usize + group * 4;
                words[offset..offset + 4].copy_from_slice(&[
                    super::super::saliency::READY,
                    group as u32,
                    edges as u32,
                    contrast as u32,
                ]);
            }
        }
        let valid = |words: &[u32]| {
            validate_artifact(bytemuck::cast_slice(words), layout, &dc, &hf, frame, None).is_ok()
        };
        assert!(valid(&words));
        if saliency {
            for group in [0, layout.saliency_groups - 1] {
                let offset = (layout.saliency_offset + group * 4) as usize;
                for field in 0..4 {
                    let mut corrupt = words.clone();
                    corrupt[offset + field] = match field {
                        0 => 0,
                        1 | 2 => words[offset + field] + 1,
                        _ => words[offset + 2] * 765 + 1,
                    };
                    assert!(!valid(&corrupt), "saliency group {group} field {field}");
                }
            }
            let mut padding = words.clone();
            *padding.last_mut().unwrap() = 1;
            assert!(!valid(&padding));
        }
        // Every new AC header field, its presence marker and final ready status.
        for index in [0, 4, 60, 61, 62, 63, 64, 65, 66, 67] {
            let mut invalid = words.clone();
            invalid[index] ^= 1;
            assert!(!valid(&invalid), "header word {index}");
        }
        for pass in 0..layout.ac_pass_count {
            for block in [
                pass * layout.strategy_len,
                (pass + 1) * layout.strategy_len - 1,
            ] {
                let mut missing = words.clone();
                missing[(layout.ac_descriptor_offset + block) as usize] = 0;
                assert!(!valid(&missing), "missing block {block}");
                let mut padding = words.clone();
                let end = layout.ac_fragment_offset + (block + 1) * layout.ac_words_per_block;
                padding[end as usize - 1] = 1;
                assert!(!valid(&padding), "block padding {block}");
            }
        }
        assert!(!valid(&words[..words.len() - 1]));
    }
}
