use super::reference::Case;

pub(super) const CASES: &[Case] = &[
    Case {
        name: "lossless_pfm",
        input_sha256: "61ae52b5851ab2e156aec1d22502e8e1ec0cdf6bc0d6a3956ac6d7d6d2969d5e",
        descriptor_sha256: "7086f95f318178163c561e375a2db06be92083a5bb0cb78cb0076a3506b6d3e9",
    },
    Case {
        name: "alpha_nonpremultiplied",
        input_sha256: "15acbe3edbfd5a75c7609726ae60526ffc812642b5dd6be8475f0b990ce9b1db",
        descriptor_sha256: "a1d53ce1679cc7ff146433feb7920a27b14a62ce7e72c07acf627fe7e7765969",
    },
    Case {
        name: "alpha_premultiplied",
        input_sha256: "5028e630dc358bf4031cbf06e8a80f2bfbfa21fd0d088af74d83b473d3dd540a",
        descriptor_sha256: "25e183fea4c96583f1d17f63e7547392f6d4f7f7047509c5ba1e42f8e1245fde",
    },
    Case {
        name: "alpha_triangles",
        input_sha256: "19ac7752a23ad2b22814064cb6b62a581b48be18ed73b5ccc2340888c114d2c9",
        descriptor_sha256: "5f1cd610e9533b589896b91c2e6f0ee87ba12a94b36f7cd51a8f92cd089073c0",
    },
    Case {
        name: "spot",
        input_sha256: "69d48f7ace25683db9eb908f112cef112bd1583870abab0fbf166018889893fa",
        descriptor_sha256: "6f2f0d6d37faa979d11fe8e1ca31bfa658563b16ca8f2212645d81fd636aac29",
    },
    Case {
        name: "sunset_logo",
        input_sha256: "6617480923e1fdef555e165a1e7df9ca648068dd0bdbc41a22c0e4213392d834",
        descriptor_sha256: "54a0324f62e3ceee867913ec976c59aaa73a70fbb9c733168cba3f0c1b2c34b9",
    },
];
