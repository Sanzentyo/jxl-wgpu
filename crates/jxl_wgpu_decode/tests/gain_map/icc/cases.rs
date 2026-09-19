use super::*;
use jxl_gpu_protocol::DisplayIntensity;
use jxl_test_support::{
    fixtures::hdr as corpus,
    oracles::{color, hdr},
};
use jxl_wgpu_decode::gain_map::GainMapRendition;

pub(super) struct Case {
    pub name: String,
    pub bytes: Vec<u8>,
    pub extent: [usize; 2],
    pub orientation: u32,
    pub rendering: GainMapRendering,
    pub pcs: Vec<[[f64; 2]; 3]>,
    pub alpha: Vec<[f64; 2]>,
}

// Independent chromaticity-derived Bradford matrix, with signed interval arithmetic.
// Keep the existing gain (2e-4) and PCS (5e-5) arithmetic contracts at their own stages.
fn pcs(center: [f64; 3], bounds: [[f64; 2]; 3], space: ColorSpace) -> [[f64; 2]; 3] {
    color::pcs_matrix(space).map(|row| {
        let value: f64 = (0..3).map(|c| row[c] * center[c]).sum();
        let interval: [f64; 2] = std::array::from_fn(|edge| {
            (0..3)
                .map(|c| row[c] * bounds[c][if row[c] >= 0.0 { edge } else { 1 - edge }])
                .sum()
        });
        [
            value,
            (value - interval[0]).max(interval[1] - value) + 5e-5 * (1.0 + value.abs()),
        ]
    })
}

pub(super) fn all() -> Vec<Case> {
    let mut cases = Vec::new();
    for line in std::fs::read_to_string(directory().join("cases.txt"))
        .unwrap()
        .lines()
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        let name = fields[0];
        let bytes = std::fs::read(directory().join(format!("{name}.jxl"))).unwrap();
        let parsed = jxl_gpu_bitstream::parse(&bytes, Default::default()).unwrap();
        let bundle = GainMapBundle::parse(
            parsed
                .auxiliary_boxes()
                .iter()
                .find(|b| b.box_type == JHGM)
                .unwrap()
                .payload,
            Default::default(),
        )
        .unwrap();
        let wide = fields[4] == "1";
        let base = floats(&format!("{name}.base.f32"));
        let linear = oracle(
            &base,
            &floats(&format!("{name}.gain.f32")),
            fields[5].parse().unwrap(),
            fields[6].parse().unwrap(),
            bundle.metadata(),
            wide,
        );
        let pcs = linear
            .as_chunks::<4>()
            .0
            .iter()
            .map(|rgba| {
                let rgb = [rgba[0], rgba[1], rgba[2]];
                pcs(
                    rgb,
                    rgb.map(|v| [v - 2e-4 * (1.0 + v.abs()), v + 2e-4 * (1.0 + v.abs())]),
                    if wide {
                        ColorSpace::Bt2020
                    } else {
                        ColorSpace::Bt709
                    },
                )
            })
            .collect();
        cases.push(Case {
            name: name.to_owned(),
            bytes,
            extent: [17, 9],
            orientation: fields[7].parse().unwrap(),
            rendering: Default::default(),
            pcs,
            alpha: base
                .as_chunks::<4>()
                .0
                .iter()
                .map(|rgba| [rgba[3] - 2e-7, rgba[3] + 2e-7])
                .collect(),
        });
    }
    assert_eq!(cases.len(), 64);
    for (index, source) in corpus::cases()
        .into_iter()
        .filter(|c| !c.sequence)
        .enumerate()
    {
        let mut map = reference::Map::load((index % 4) * 4);
        let bytes = source.bytes();
        let inventory = jxl_gpu_bitstream::parse(&bytes, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        source.validate(&inventory);
        let original = source.reference(false);
        let linear: Vec<_> = if source.xyb {
            source.reference(true).into_iter().map(f64::from).collect()
        } else {
            original
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|rgba| {
                    let rgb = hdr::to_linear(
                        [rgba[0], rgba[1], rgba[2]].map(f64::from),
                        source.transfer,
                        source.space,
                        source.nits,
                    );
                    [rgb[0], rgb[1], rgb[2], f64::from(rgba[3])]
                })
                .collect()
        };
        for reverse in [false, true] {
            map.metadata.base_hdr_headroom.numerator = if reverse { 3 } else { 1 };
            map.metadata.alternate_hdr_headroom.numerator = if reverse { 1 } else { 3 };
            map.metadata.base_hdr_headroom.denominator = 1;
            map.metadata.alternate_hdr_headroom.denominator = 1;
            let white = [100.0_f32, 203.0, 500.0][index % 3];
            let reference = reference::Reference {
                map: &map,
                extent: [source.width, source.height],
                weight: if reverse { -0.5 } else { 0.5 },
                scale: source.nits / f64::from(white),
            };
            let pcs = linear
                .as_chunks::<4>()
                .0
                .iter()
                .enumerate()
                .map(|(p, rgba)| {
                    let rgb = [rgba[0], rgba[1], rgba[2]];
                    let interval = if source.xyb {
                        rgb.map(|v| {
                            [
                                v - f64::from(source.tolerance()) * (1.0 + v.abs()),
                                v + f64::from(source.tolerance()) * (1.0 + v.abs()),
                            ]
                        })
                    } else {
                        hdr::linear_interval(
                            [original[p * 4], original[p * 4 + 1], original[p * 4 + 2]]
                                .map(f64::from),
                            source.transfer,
                            source.space,
                            source.nits,
                            f64::from(source.tolerance()),
                        )
                    };
                    let xy = [p % source.width, p / source.width];
                    let center = reference.rgb(rgb, xy);
                    let interval = reference.interval(interval, xy);
                    let bounds = std::array::from_fn(|c| {
                        [
                            interval[c][0] - 2e-4 * (1.0 + center[c].abs()),
                            interval[c][1] + 2e-4 * (1.0 + center[c].abs()),
                        ]
                    });
                    pcs(center, bounds, source.space)
                })
                .collect();
            cases.push(Case {
                name: format!("{}-reverse-{reverse}", source.name),
                bytes: map.container(&bytes),
                extent: [source.width, source.height],
                orientation: inventory.image_header.orientation,
                rendering: GainMapRendering {
                    rendition: GainMapRendition::DisplayHeadroom(2.0),
                    reference_white: DisplayIntensity::new(white).unwrap(),
                },
                pcs,
                alpha: linear
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|rgba| [rgba[3] - 2e-6, rgba[3] + 2e-6])
                    .collect(),
            });
        }
    }
    cases
}
