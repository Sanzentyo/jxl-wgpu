use super::{
    HuffmanTable as Huffman, JpegReconstructionError as Error, JpegScan as Scan,
    QuantizationTable as Quant,
};
type Result<T> = std::result::Result<T, Error>;

pub(super) fn validate_huffman_counts(counts: &[u16; 17]) -> Result<()> {
    if counts[0] != 0 {
        return Err(Error::Invalid("zero-length Huffman code"));
    }
    let mut available = 1u32;
    for &count in &counts[1..] {
        available = (available * 2)
            .checked_sub(u32::from(count))
            .ok_or(Error::Invalid("oversubscribed Huffman code"))?;
    }
    // Counting the synthetic terminal prevents real symbols from exhausting a bit
    // length and receiving the forbidden all-one code. Incomplete trees remain legal.
    Ok(())
}

pub(super) fn validate_quant_groups(markers: &[u8], tables: &[Quant]) -> Result<()> {
    let mut table = 0;
    for &marker in markers {
        if marker != 0xdb {
            continue;
        }
        loop {
            let q = tables.get(table).ok_or(Error::Invalid("DQT group count"))?;
            table += 1;
            if q.last {
                break;
            }
        }
    }
    if table != tables.len() {
        return Err(Error::Invalid("unconsumed quantization table"));
    }
    Ok(())
}

pub(super) fn validate_huffman_use(
    markers: &[u8],
    tables: &[Huffman],
    scans: &[Scan],
) -> Result<()> {
    let (mut table, mut scan, mut progressive) = (0, 0, false);
    let (mut dc, mut ac) = ([false; 4], [false; 4]);
    for &marker in markers {
        match marker {
            0xc2 => progressive = true,
            0xc4 => {
                let first = table;
                loop {
                    let h = tables.get(table).ok_or(Error::Invalid("DHT group count"))?;
                    table += 1;
                    if h.values.is_empty() {
                        if table != first + 1 {
                            return Err(Error::Invalid("empty DHT within nonempty group"));
                        }
                        // An empty marker ends its group regardless of the last flag and
                        // neither defines a table nor clears an earlier definition.
                        break;
                    }
                    if h.ac {
                        ac[h.index as usize] = true;
                    } else {
                        dc[h.index as usize] = true;
                    }
                    if h.last {
                        break;
                    }
                }
            }
            0xda => {
                let s = &scans[scan];
                scan += 1;
                for c in &s.components {
                    if (!progressive || (s.start == 0 && s.high == 0)) && !dc[c.dc as usize] {
                        return Err(Error::Invalid("DC table used before definition"));
                    }
                    if (!progressive || s.start != 0) && !ac[c.ac as usize] {
                        return Err(Error::Invalid("AC table used before definition"));
                    }
                }
            }
            _ => {}
        }
    }
    if table != tables.len() {
        return Err(Error::Invalid("unconsumed Huffman table"));
    }
    Ok(())
}
