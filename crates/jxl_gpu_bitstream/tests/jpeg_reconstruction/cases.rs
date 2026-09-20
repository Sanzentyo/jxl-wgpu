// Input identities are pinned before exercising the production parser/emitter.
use sha2::{Digest, Sha256};

pub struct Case {
    pub name: &'static str,
    pub input: &'static [u8],
    pub jpeg: &'static [u8],
    pub input_sha256: &'static str,
    pub jpeg_sha256: &'static str,
    pub scans: usize,
    pub padding_bits: u32,
    pub official: bool,
}

impl Case {
    pub fn validate(&self) {
        for (bytes, expected) in [
            (self.input, self.input_sha256),
            (self.jpeg, self.jpeg_sha256),
        ] {
            let expected: Vec<_> = (0..expected.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&expected[i..i + 2], 16).unwrap())
                .collect();
            assert_eq!(
                &Sha256::digest(bytes)[..],
                expected,
                "{} identity",
                self.name
            );
        }
    }
}

pub const CASES: &[Case] = &[
    Case {
        name: "bench_oriented_brg",
        input: include_bytes!(
            "../../../jxl_wgpu_decode/test-data/official_conformance/bench_oriented_brg/input.jxl"
        ),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/bench_oriented_brg/source.jpg"),
        input_sha256: "e223fed907c6238622b2b6ec1c80609050c9d2db4d759cf3ca6f0db304cbb82a",
        jpeg_sha256: "cad665c67d74e3e5cf775ef618c73b0e70dfece33db7dbe0130bc889f2214e1b",
        scans: 1,
        padding_bits: 0,
        official: true,
    },
    Case {
        name: "cafe",
        input: include_bytes!(
            "../../../jxl_wgpu_decode/test-data/official_conformance/cafe/input.jxl"
        ),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/cafe/source.jpg"),
        input_sha256: "cb4603bd089483fc1d63a039a3b758ec510f0090b74ce245b88b1e8feb54f0eb",
        jpeg_sha256: "b6f6e4f820ac69234184434e5b77156401fb782bb46b96e26255ff51be1ec290",
        scans: 1,
        padding_bits: 0,
        official: true,
    },
    Case {
        name: "grayscale_jpeg",
        input: include_bytes!(
            "../../../jxl_wgpu_decode/test-data/official_conformance/grayscale_jpeg/input.jxl"
        ),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/grayscale_jpeg/source.jpg"),
        input_sha256: "346cc92b55864b0efb9f85cdddd02b586473ff18b03718e5a02fa03cb3e86844",
        jpeg_sha256: "a170600cc02b2b029dc79c5ee72dbf9107e8de5a64f1d8b1732c759e09a3d41d",
        scans: 1,
        padding_bits: 0,
        official: true,
    },
    Case {
        name: "gray_restart",
        input: include_bytes!("../../test-data/jpeg_reconstruction/gray_restart/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/gray_restart/source.jpg"),
        input_sha256: "7a276e69fc21f192ad9f374cb8457eb759b5aac82c4b64ae091875c206ed2d43",
        jpeg_sha256: "2babe38cba6116718518a2fa1a4bfaca7ff88abf28972f40f76ea5a70d5623f1",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "gray_progressive_restart",
        input: include_bytes!(
            "../../test-data/jpeg_reconstruction/gray_progressive_restart/input.jxl"
        ),
        jpeg: include_bytes!(
            "../../test-data/jpeg_reconstruction/gray_progressive_restart/source.jpg"
        ),
        input_sha256: "1ef4c9c8147bb1308ef9b46df739f9e9f94951462819bc8c0dd926afe1a8936f",
        jpeg_sha256: "b401ec08b84b4bf53653e26bbbee51f6439f8573ce0a1db78981eacb14b5b5a4",
        scans: 6,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "rgb_sequential",
        input: include_bytes!("../../test-data/jpeg_reconstruction/rgb_sequential/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/rgb_sequential/source.jpg"),
        input_sha256: "0578b7b1016201322d42354c2e3a17164fd8cdb24efc59641b22c2aa53d34ee4",
        jpeg_sha256: "ae825dd55c7bb9339dd77da943ff064de85c9c4a57d352c4ee5477b26be0cc72",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "rgb_progressive",
        input: include_bytes!("../../test-data/jpeg_reconstruction/rgb_progressive/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/rgb_progressive/source.jpg"),
        input_sha256: "5fb960d376b70cfaa5153f901c32f737bb4e8782f7a93bd9b39c9a4edd49009e",
        jpeg_sha256: "68b43a8f555e251060381de53145ddfedbdd0e7245b0b551b82a22fe5c277fb2",
        scans: 14,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "ycbcr444_sequential",
        input: include_bytes!("../../test-data/jpeg_reconstruction/ycbcr444_sequential/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/ycbcr444_sequential/source.jpg"),
        input_sha256: "eeb5fa696ff2b88ab662cf51c83ec5462bd8fa35d1a309abc087ca15f1133e16",
        jpeg_sha256: "c3d6a12db2b753d653e07e7c32a4ef3665d4b202121cf3a197a3d0164712b7e6",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "ycbcr444_progressive",
        input: include_bytes!("../../test-data/jpeg_reconstruction/ycbcr444_progressive/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/ycbcr444_progressive/source.jpg"),
        input_sha256: "4ba834392b6b024c377f98e3a84e6dc6c5d318988bbc7002ed8649ea58fd059f",
        jpeg_sha256: "66f17ad4bf0ec72575e1c309191cfd0084cef4f72b2f6fc65d6e184c3a101100",
        scans: 10,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "ycbcr420_sequential",
        input: include_bytes!("../../test-data/jpeg_reconstruction/ycbcr420_sequential/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/ycbcr420_sequential/source.jpg"),
        input_sha256: "b047a264240b303da7e5799af3b8d5352767e39853d17ac0cdf545a386822ccb",
        jpeg_sha256: "a9a7ef8df3959c23b28be3856b5302ac2195f7fbfa9058f368e9c4ed07b711e7",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "ycbcr420_progressive_restart",
        input: include_bytes!(
            "../../test-data/jpeg_reconstruction/ycbcr420_progressive_restart/input.jxl"
        ),
        jpeg: include_bytes!(
            "../../test-data/jpeg_reconstruction/ycbcr420_progressive_restart/source.jpg"
        ),
        input_sha256: "cd85b63627f67d48d60a81b9fed952343669b2993ab0d28a70686af939204006",
        jpeg_sha256: "6250918ec39b4a181960c10239eed4ad1b4fbc71cd118295a846c51fb7ccb717",
        scans: 10,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "ycbcr440_progressive_restart",
        input: include_bytes!(
            "../../test-data/jpeg_reconstruction/ycbcr440_progressive_restart/input.jxl"
        ),
        jpeg: include_bytes!(
            "../../test-data/jpeg_reconstruction/ycbcr440_progressive_restart/source.jpg"
        ),
        input_sha256: "a5c86d76933ca65ef662201236208241066448cdf3715b3c3aca940f95fde4c3",
        jpeg_sha256: "b23f58bac9bc043e1cbb5ea8f749df03fbd2088cbb043631bd05dc3939050ef4",
        scans: 10,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "empty_dht",
        input: include_bytes!("../../test-data/jpeg_reconstruction/empty_dht/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/empty_dht/source.jpg"),
        input_sha256: "9246e9a5eb35dda07eeb63a8eaae873c699945b86e78a7338853bc05af68ca58",
        jpeg_sha256: "e0532338f4208cb1c27595dfa80d24f5a5bdbc21a75a8e5b6fa2fa20f4c79cd6",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "merged_dht",
        input: include_bytes!("../../test-data/jpeg_reconstruction/merged_dht/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/merged_dht/source.jpg"),
        input_sha256: "e3d79ff392975e163b22e7f55537f5fc80d4540948a11b337fd1a8fec552cbc1",
        jpeg_sha256: "ac7d12073fad0ef352b0f692b61caed0454fa1ce0aea88a443f018c4c218317f",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "unused_quant",
        input: include_bytes!("../../test-data/jpeg_reconstruction/unused_quant/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/unused_quant/source.jpg"),
        input_sha256: "397e3ff4988db2b926ac89f2d347135745f6d34d9f6ab2ed79b8568da92dc2f5",
        jpeg_sha256: "c04bdd4f5308664d910a746644942396a1917bf2202b787210df36a6d399ab3a",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "custom_selectors",
        input: include_bytes!("../../test-data/jpeg_reconstruction/custom_selectors/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/custom_selectors/source.jpg"),
        input_sha256: "7b3c9fd56450652a1533b45f2648e6fb5b378735205f9f72aec83a90c88124e5",
        jpeg_sha256: "6c3eaa5a2bd5e3dbfa1f068f53b0bb36db918e9033b6bdf30c2963ee36b1499c",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "quant16",
        input: include_bytes!("../../test-data/jpeg_reconstruction/quant16/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/quant16/source.jpg"),
        input_sha256: "5f13b7b1a1127b66c27bb0fdf88e330b36af554e6d5dd1daa30b365615cb9ed0",
        jpeg_sha256: "9b702d5789d8ec98c501e9dee30104115cdd2cd76c9f7614a3558fd89b85d915",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "icc_chunks",
        input: include_bytes!("../../test-data/jpeg_reconstruction/icc_chunks/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/icc_chunks/source.jpg"),
        input_sha256: "19fe4eb616ed7f84c9f9d186db6989730f9d792ddf7dba80c5ddf401115eda94",
        jpeg_sha256: "6eef858cd6c4522581c4a7a1792ec9c9738b99e437cc46e5b653bcb8cbe4745d",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "exif_rotated",
        input: include_bytes!("../../test-data/jpeg_reconstruction/exif_rotated/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/exif_rotated/source.jpg"),
        input_sha256: "52430ddecdcbcb784811e2c2598dbfa92edd52a31e9828046c48ca0b83fe7192",
        jpeg_sha256: "b3effe3830995e2206f8319443ecf38d50e9ff390ede82370753051c9f0f935e",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "xmp_comment",
        input: include_bytes!("../../test-data/jpeg_reconstruction/xmp_comment/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/xmp_comment/source.jpg"),
        input_sha256: "fc2610177d0570eb46e6809443e04c5ea5b3b3dea287750394fadf6faaac4b0f",
        jpeg_sha256: "136229c5f33d202b5da804274f914415319fb7c08ec06f8ec728dc76d98151c1",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "metadata_combined",
        input: include_bytes!("../../test-data/jpeg_reconstruction/metadata_combined/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/metadata_combined/source.jpg"),
        input_sha256: "fa8c8d31860b9544524bc6ce7c87ff85763dc383f7043a5c04b93fb42569c8f1",
        jpeg_sha256: "c10457f2235cb2b73c92a53f8453fe723888ee8181215541fb47f387be46d7fb",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "marker_fill",
        input: include_bytes!("../../test-data/jpeg_reconstruction/marker_fill/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/marker_fill/source.jpg"),
        input_sha256: "1eaf4e7e3740824ede6aa6d89a3cbd81bca04c6a4016dec49dfc7e4f527d79be",
        jpeg_sha256: "96dae778409e1a7895b51b6d383f249e136fd6d45207d7b6fdccdfcd8a746a9c",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "tail_bytes",
        input: include_bytes!("../../test-data/jpeg_reconstruction/tail_bytes/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/tail_bytes/source.jpg"),
        input_sha256: "a827b3190d278d4b826fb5f878ddb89fd7b0e98694ed770fdf61a2c83a8b0a1a",
        jpeg_sha256: "b132285b10fdbc60e362bd7d005353bbb153ecf43f9b4c11d24a3cdb7bd8db93",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "merged_ycbcr",
        input: include_bytes!("../../test-data/jpeg_reconstruction/merged_ycbcr/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/merged_ycbcr/source.jpg"),
        input_sha256: "b403bfa546d96ee0f43442cf15afed488239bc0d1d5090066a09fd476cc0bfc3",
        jpeg_sha256: "b65e9ecde266c73ee15ec34a4b97246de9ed15cba7fc5fcdc7b9d4dbcc5e448f",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "gray_zero_padding",
        input: include_bytes!("../../test-data/jpeg_reconstruction/gray_zero_padding/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/gray_zero_padding/source.jpg"),
        input_sha256: "763f43fec9801413484e3c7b031bc6eb491790f898b24b699774f9ee40e98a27",
        jpeg_sha256: "5fc0e599ffc7ef544cba854e43530c320a8bb527bf1ed03af1a80fec1b15897a",
        scans: 1,
        padding_bits: 150,
        official: false,
    },
    Case {
        name: "gray_mixed_padding",
        input: include_bytes!("../../test-data/jpeg_reconstruction/gray_mixed_padding/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/gray_mixed_padding/source.jpg"),
        input_sha256: "d4032f05f5b59dc15b167713d8d86da90e518b659fd5c541cfd86b89ea9f16e0",
        jpeg_sha256: "c3d2dd1df69568e755500f0c12766d77566ad485a7cf0e0cf1db01a170208880",
        scans: 1,
        padding_bits: 150,
        official: false,
    },
    Case {
        name: "gray_extra_zrl1",
        input: include_bytes!("../../test-data/jpeg_reconstruction/gray_extra_zrl1/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/gray_extra_zrl1/source.jpg"),
        input_sha256: "91b0f4e795f494c6df21ee13e27e063cc6c8c75b2cd58cbfdadc73740c0138ae",
        jpeg_sha256: "1129e1309cb845090696b71766eaa598ce64fe0ea9fe86def1152bb2769356c2",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "gray_extra_zrl2",
        input: include_bytes!("../../test-data/jpeg_reconstruction/gray_extra_zrl2/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/gray_extra_zrl2/source.jpg"),
        input_sha256: "f9b1acb0e807efaa94c654964d083a876c301ff2f0b4341035cbd540e85bf157",
        jpeg_sha256: "342a4b0c8c6bf39a8e7377df9942c775c49e17d46b45b7923c9775d0b5446888",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "gray_extra_zrl3",
        input: include_bytes!("../../test-data/jpeg_reconstruction/gray_extra_zrl3/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/gray_extra_zrl3/source.jpg"),
        input_sha256: "45f1f8331c4b0f37ce9e748c698ad07115f5efaa2cff99424d835b6ab3abb735",
        jpeg_sha256: "57fc1ff2421e7fcfabefe9903e1753b8ef2fd46d7232daa89c1d3cd81eeae2e1",
        scans: 1,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "gray_long_eob",
        input: include_bytes!("../../test-data/jpeg_reconstruction/gray_long_eob/input.jxl"),
        jpeg: include_bytes!("../../test-data/jpeg_reconstruction/gray_long_eob/source.jpg"),
        input_sha256: "f7ccb738c2b3e8ad9e62e2051401f0cac173047ed6f5fc3fd3f4845459dc0537",
        jpeg_sha256: "e7fdf25aed30242253141303b29e1eb8f1f601fd56b612b8534096cda5ec3d5b",
        scans: 6,
        padding_bits: 0,
        official: false,
    },
    Case {
        name: "gray_progressive_restart_zero_padding",
        input: include_bytes!(
            "../../test-data/jpeg_reconstruction/gray_progressive_restart_zero_padding/input.jxl"
        ),
        jpeg: include_bytes!(
            "../../test-data/jpeg_reconstruction/gray_progressive_restart_zero_padding/source.jpg"
        ),
        input_sha256: "7a53774b8058dcefcb26123be08891d31aea7d85070844b72897b1285c1e03f2",
        jpeg_sha256: "7c6099b3fa3bd8c16205ce705eef2151bcf3579b72245209f7d9147c100c415d",
        scans: 6,
        padding_bits: 1076,
        official: false,
    },
    Case {
        name: "gray_progressive_restart_mixed_padding",
        input: include_bytes!(
            "../../test-data/jpeg_reconstruction/gray_progressive_restart_mixed_padding/input.jxl"
        ),
        jpeg: include_bytes!(
            "../../test-data/jpeg_reconstruction/gray_progressive_restart_mixed_padding/source.jpg"
        ),
        input_sha256: "4fde14b5a05904b090f61bcd980491dc8e569aa140eb8503f5150500ba024a4c",
        jpeg_sha256: "62b5617b33dad64c28e276f944ec0f83af4b937ceda14f970bc832f06c0eec1f",
        scans: 6,
        padding_bits: 1076,
        official: false,
    },
    Case {
        name: "rgb_progressive_zero_padding",
        input: include_bytes!(
            "../../test-data/jpeg_reconstruction/rgb_progressive_zero_padding/input.jxl"
        ),
        jpeg: include_bytes!(
            "../../test-data/jpeg_reconstruction/rgb_progressive_zero_padding/source.jpg"
        ),
        input_sha256: "a25cf1410e6373eda497f2cb2b1da2269dedd5b562c26edeb3e8fde812613aa7",
        jpeg_sha256: "81a3c95bb0efd3f1451760b4e6f1c3fe399fa1863586feb74e37e01ec3746ba4",
        scans: 14,
        padding_bits: 63,
        official: false,
    },
    Case {
        name: "rgb_progressive_mixed_padding",
        input: include_bytes!(
            "../../test-data/jpeg_reconstruction/rgb_progressive_mixed_padding/input.jxl"
        ),
        jpeg: include_bytes!(
            "../../test-data/jpeg_reconstruction/rgb_progressive_mixed_padding/source.jpg"
        ),
        input_sha256: "38686a43ffb7000f3fab4c5feda800632523557d7674be81bc84185f73ba69a9",
        jpeg_sha256: "4064ef166781a6ce28ebb9853db1e67626aa79ede680ad774344e06459da49d8",
        scans: 14,
        padding_bits: 63,
        official: false,
    },
    Case {
        name: "ycbcr420_progressive_restart_zero_padding",
        input: include_bytes!(
            "../../test-data/jpeg_reconstruction/ycbcr420_progressive_restart_zero_padding/input.jxl"
        ),
        jpeg: include_bytes!(
            "../../test-data/jpeg_reconstruction/ycbcr420_progressive_restart_zero_padding/source.jpg"
        ),
        input_sha256: "cb6008bc37a047171459e0ce9303cf89be2d1635fc3ad1b6987aa264f56d0241",
        jpeg_sha256: "23e740d635742adebd414d479b6874a1737954531a44940fec0a60481a862089",
        scans: 10,
        padding_bits: 318,
        official: false,
    },
    Case {
        name: "ycbcr420_progressive_restart_mixed_padding",
        input: include_bytes!(
            "../../test-data/jpeg_reconstruction/ycbcr420_progressive_restart_mixed_padding/input.jxl"
        ),
        jpeg: include_bytes!(
            "../../test-data/jpeg_reconstruction/ycbcr420_progressive_restart_mixed_padding/source.jpg"
        ),
        input_sha256: "c1dbbf3ce1aaa8e0dc3d481387fbda8d9a76557c1377f6af97548c0a4296d720",
        jpeg_sha256: "23f30161b34042632704ddc12efd2f1fa31d8194993326e15769e4bcf3d48054",
        scans: 10,
        padding_bits: 318,
        official: false,
    },
];
