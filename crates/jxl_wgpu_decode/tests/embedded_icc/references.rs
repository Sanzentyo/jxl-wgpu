use std::path::Path;

pub(super) fn read(directory: &Path, name: &str) -> Vec<[f32; 6]> {
    let bytes = std::fs::read(directory.join(format!("decoder/{name}.reference"))).unwrap();
    let (records, tail) = bytes.as_chunks::<28>();
    assert!(tail.is_empty());
    records
        .iter()
        .map(|record| {
            let values = std::array::from_fn(|c| {
                f32::from_le_bytes(record[c * 4..c * 4 + 4].try_into().unwrap())
            });
            let [native, exact, lower, upper, native_lower, native_upper] = values;
            assert!(values.iter().all(|v| v.is_finite()));
            assert!(lower <= exact && exact <= upper);
            assert!(native_lower <= native && native <= native_upper);
            assert_eq!(u32::from_le_bytes(record[24..].try_into().unwrap()), 0);
            values
        })
        .collect()
}
