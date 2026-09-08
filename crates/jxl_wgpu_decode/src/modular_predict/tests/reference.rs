//! Native-width mathematical oracle, independent of the shader's word-pair arithmetic.

pub(super) fn predict(data: &[u32; 40]) -> Vec<u32> {
    let [n, w, nw, ne, nn, ww, nee, sample] = std::array::from_fn(|i| i64::from(data[i] as i32));
    let [te_w, te_nw, te_n, te_ne] = std::array::from_fn(|i| i64::from(data[8 + i] as i32));
    let coefficients: [i64; 7] = std::array::from_fn(|i| i64::from(data[24 + i]));
    let [p1, p2, p3a, p3b, p3c, p3d, p3e] = coefficients;
    let [n3, w3, nw3, ne3, nn3] = [n, w, nw, ne, nn].map(|v| v * 8);
    let subpred = [
        w3 + ne3 - n3,
        n3 - (((te_w + te_n + te_ne) * p1) >> 5),
        w3 - (((te_w + te_n + te_nw) * p2) >> 5),
        n3 - ((te_nw * p3a + te_n * p3b + te_ne * p3c + (nn3 - n3) * p3d + (nw3 - w3) * p3e) >> 5),
    ];
    let mut weights: [u32; 4] = std::array::from_fn(|c| {
        let error = data[12 + c]
            .wrapping_add(data[16 + c])
            .wrapping_add(data[20 + c]);
        let shift = (u64::from(error) + 1).ilog2().saturating_sub(5);
        4 + ((data[31 + c] * ((1u32 << 24) / ((error >> shift) + 1))) >> shift)
    });
    let shift = weights.iter().sum::<u32>().ilog2() - 4;
    weights.iter_mut().for_each(|w| *w >>= shift);
    let sum_weights = weights.iter().sum::<u32>();
    let sum = i64::from((sum_weights >> 1) - 1)
        + subpred
            .iter()
            .zip(weights)
            .map(|(p, w)| p * i64::from(w))
            .sum::<i64>();
    let mut prediction = (sum * i64::from((1u32 << 24) / sum_weights)) >> 24;
    if ((te_n ^ te_w) | (te_n ^ te_nw)) <= 0 {
        prediction = prediction.clamp(n3.min(w3).min(ne3), n3.max(w3).max(ne3));
    }
    let max_error = [te_n, te_nw, te_ne].into_iter().fold(te_w, |max, error| {
        if error.abs() > max.abs() { error } else { max }
    });
    let mut output = super::words(prediction).to_vec();
    output.push(max_error as u32);
    for p in subpred {
        output.extend(super::words(p));
    }
    let predictors = [
        0,
        w,
        n,
        (w + n) / 2,
        if (n - nw).abs() < (w - nw).abs() {
            w
        } else {
            n
        },
        (n + w - nw).clamp(n.min(w), n.max(w)),
        (prediction + 3) >> 3,
        ne,
        nw,
        ww,
        (w + nw) / 2,
        (n + nw) / 2,
        (n + ne) / 2,
        (6 * n - 2 * nn + 7 * w + ww + nee + 3 * ne + 8) / 16,
    ];
    output.extend(predictors.map(|v| v as u32));
    output.push((prediction - sample * 8) as u32);
    output.extend(subpred.map(|p| (((p - sample * 8).abs() + 3) >> 3) as u32));
    output
}
