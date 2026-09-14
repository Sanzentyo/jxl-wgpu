use super::{apply, interval};

#[test]
fn intervals_enclose_cube_faces_and_black_boundary() {
    let weights = [0.212639005871510, 0.715168678767756, 0.072192315360734];
    for r in [-2.0, -0.2, 0.0, 0.9, 1.0, 1.2, 4.0] {
        for g in [-1.0, 0.0, 0.3, 1.0, 3.0] {
            for b in [-0.5, 0.0, 0.5, 1.0, 2.0] {
                let rgb = [r, g, b];
                for p in [0.0, 0.1, 0.5, 0.9, 1.0] {
                    let bounds = interval(rgb.map(|c| [c - 0.01, c + 0.01]), weights, p);
                    for sample in 0..27 {
                        let input = std::array::from_fn(|c| {
                            rgb[c] + ((sample / 3usize.pow(c as u32)) % 3) as f64 * 0.01 - 0.01
                        });
                        let actual = apply(input, weights, p);
                        for c in 0..3 {
                            assert!(
                                actual[c] >= bounds[c][0] - 1e-12
                                    && actual[c] <= bounds[c][1] + 1e-12,
                                "{input:?}/{p}: {actual:?} in {bounds:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}
