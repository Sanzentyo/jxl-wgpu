//! Native Modular YCbCr sampling cases shared by regeneration and GPU conformance.

use jxl_gpu_bitstream::{
    CodestreamInventory, EdgePreservingFilterInventory, ExtraChannelTypeInventory, FrameEncoding,
    GaborishInventory, RestorationFilterInventory, SampleBitDepth,
};

mod transforms;
pub use transforms::{NativeTopology, Squeeze, Transform};

#[derive(Clone, Debug)]
pub struct Case {
    pub name: String,
    pub selectors: [u32; 3],
    pub size: [u32; 2],
    pub bit_depth: SampleBitDepth,
    pub grayscale: bool,
    pub gaborish: bool,
    pub epf_iterations: u32,
    pub upsampling: u32,
    pub extra_factors: Vec<u32>,
    pub associated: bool,
    pub orientation: u32,
    pub group_size_shift: u32,
    pub passes: u32,
    /// First published boundary when progressive output is requested; empty leading passes vanish.
    pub first_output_pass: u32,
    pub transforms: Vec<Transform>,
}

impl Case {
    fn new(name: String) -> Self {
        Self {
            name,
            selectors: [0, 1, 0],
            size: [37, 19],
            bit_depth: SampleBitDepth::Integer {
                bits_per_sample: 16,
            },
            grayscale: false,
            gaborish: false,
            epf_iterations: 0,
            upsampling: 1,
            extra_factors: Vec::new(),
            associated: false,
            orientation: 1,
            group_size_shift: 1,
            passes: 1,
            first_output_pass: 0,
            transforms: Vec::new(),
        }
    }

    /// Verify the independently serialized stream actually exercises its named features.
    pub fn validate(&self, inventory: &CodestreamInventory) {
        let image = &inventory.image_header;
        assert_eq!([image.width, image.height], self.size, "{}", self.name);
        assert_eq!(image.bit_depth, self.bit_depth, "{}", self.name);
        assert_eq!(image.grayscale, self.grayscale, "{}", self.name);
        assert_eq!(image.orientation, self.orientation, "{}", self.name);
        assert!(!image.xyb_encoded);
        assert_eq!(image.extra_channels.len(), self.extra_factors.len());
        if !self.extra_factors.is_empty() {
            assert_eq!(image.extra_channels.len(), 2);
            let alpha = &image.extra_channels[0];
            assert_eq!(
                alpha.channel_type,
                ExtraChannelTypeInventory::Alpha {
                    associated: self.associated
                }
            );
            assert_eq!(
                alpha.bit_depth,
                SampleBitDepth::Integer {
                    bits_per_sample: 12
                }
            );
            let depth = &image.extra_channels[1];
            assert_eq!(depth.channel_type, ExtraChannelTypeInventory::Depth);
            assert_eq!(
                depth.bit_depth,
                SampleBitDepth::Float {
                    bits_per_sample: 32,
                    exponent_bits_per_sample: 8
                }
            );
        }
        assert_eq!(inventory.frames.len(), 1);
        let frame = &inventory.frames[0];
        assert_eq!(frame.encoding, FrameEncoding::Modular);
        assert!(frame.do_ycbcr);
        assert_eq!(frame.group_size_shift, self.group_size_shift);
        assert_eq!(frame.num_passes, self.passes);
        assert_eq!(frame.jpeg_upsampling, self.selectors, "{}", self.name);
        assert_eq!(frame.upsampling, self.upsampling, "{}", self.name);
        assert_eq!(
            frame.extra_channel_upsampling, self.extra_factors,
            "{}",
            self.name
        );
        let (gaborish, epf) = match frame.restoration_filter {
            RestorationFilterInventory::Default => (true, 2),
            RestorationFilterInventory::Custom { gaborish, epf } => (
                gaborish != GaborishInventory::Disabled,
                match epf {
                    EdgePreservingFilterInventory::Disabled => 0,
                    EdgePreservingFilterInventory::Enabled { iterations, .. } => iterations,
                },
            ),
        };
        assert_eq!(
            (gaborish, epf),
            (self.gaborish, self.epf_iterations),
            "{}",
            self.name
        );
    }
}

pub fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for cb in 0..4 {
        for y in 0..4 {
            for cr in 0..4 {
                let mut case = Case::new(format!("sampling_{cb}{y}{cr}"));
                case.selectors = [cb, y, cr];
                cases.push(case);
            }
        }
    }
    for bits_per_sample in [8, 12, 31] {
        let mut case = Case::new(format!("integer_{bits_per_sample}"));
        case.bit_depth = SampleBitDepth::Integer { bits_per_sample };
        cases.push(case);
    }
    for (bits_per_sample, exponent_bits_per_sample) in [(16, 5), (24, 7), (32, 8)] {
        let mut case = Case::new(format!("float_{bits_per_sample}"));
        case.bit_depth = SampleBitDepth::Float {
            bits_per_sample,
            exponent_bits_per_sample,
        };
        cases.push(case);
    }
    for (name, size) in [("thin_horizontal", [37, 1]), ("thin_vertical", [1, 19])] {
        let mut case = Case::new(name.into());
        case.size = size;
        cases.push(case);
    }
    let mut gray = Case::new("gray".into());
    gray.grayscale = true;
    cases.push(gray);
    for iterations in [1, 2, 3] {
        let mut case = Case::new(format!("restoration_{iterations}"));
        case.selectors = [1, 2, 3];
        case.gaborish = true;
        case.epf_iterations = iterations;
        cases.push(case);
    }
    for factor in [2, 4, 8] {
        let mut case = Case::new(format!("resampling_{factor}"));
        case.size = [53, 35];
        case.bit_depth = SampleBitDepth::Float {
            bits_per_sample: 32,
            exponent_bits_per_sample: 8,
        };
        case.upsampling = factor;
        case.extra_factors = vec![factor, 8];
        case.gaborish = true;
        case.epf_iterations = 2;
        cases.push(case);
    }
    let mut associated = Case::new("associated".into());
    associated.bit_depth = SampleBitDepth::Float {
        bits_per_sample: 32,
        exponent_bits_per_sample: 8,
    };
    associated.associated = true;
    associated.extra_factors = vec![1, 1];
    cases.push(associated.clone());
    for orientation in 2..=8 {
        let mut case = associated.clone();
        case.name = format!("orientation_{orientation}");
        case.orientation = orientation;
        cases.push(case);
    }
    for shift in 0..=3 {
        for vertical in [false, true] {
            let direction = if vertical { "vertical" } else { "horizontal" };
            let mut case = Case::new(format!("groups_{}_{direction}", 128 << shift));
            case.group_size_shift = shift;
            case.size[usize::from(vertical)] = (256 << shift) + 3;
            cases.push(case);
        }
    }
    let mut prefix = Case::new("global_prefix".into());
    prefix.size[0] = 257;
    cases.push(prefix.clone());
    prefix.name = "passes_global_prefix".into();
    prefix.passes = 2;
    cases.push(prefix);
    for directional in [false, true] {
        let mut case = Case::new(
            if directional {
                "passes_directional"
            } else {
                "passes_420"
            }
            .into(),
        );
        case.size[0] = 259;
        case.group_size_shift = 0;
        case.passes = 2;
        case.first_output_pass = if directional { 2 } else { 1 };
        if directional {
            case.selectors = [1, 2, 3];
        }
        cases.push(case);
    }
    let mut lf = Case::new("lf_extras".into());
    lf.size[0] = 2051;
    lf.group_size_shift = 0;
    lf.bit_depth = SampleBitDepth::Float {
        bits_per_sample: 32,
        exponent_bits_per_sample: 8,
    };
    lf.extra_factors = vec![8, 8];
    cases.push(lf);
    transforms::extend(&mut cases);
    cases
}
