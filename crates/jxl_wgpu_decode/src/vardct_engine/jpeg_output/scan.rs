//! Bounded scan scheduling from immutable metadata and a checked resident coefficient layout.
use super::super::jpeg::{JpegCoefficientLayout, JpegCoefficientPlane};
use super::{Error, JpegReconstructionLimits, Result, allocate, check};
use jxl_gpu_bitstream::jpeg_reconstruction::{
    JpegReconstructionMetadata as Header, JpegScan as Scan,
};

struct Geometry<'a> {
    planes: &'a [JpegCoefficientPlane],
    components: usize,
    mcus: [u32; 2],
}

#[derive(Debug)]
pub(super) struct PlannedScan {
    pub parameters: u32,
    pub tasks: Vec<[u32; 12]>,
    pub tables: Vec<u32>,
}

#[derive(Debug)]
pub(super) struct Plan {
    pub scans: Vec<PlannedScan>,
    pub masks: [[u32; 64]; 3],
    pub owned_bytes: u64,
}

fn scan_extent(scan: &Scan, geometry: &Geometry, progressive: bool) -> Result<([u32; 2], u32)> {
    if scan.components.is_empty() || scan.components.len() > geometry.components {
        return Err(Error::Invalid("scan components"));
    }
    let mut seen = [false; 3];
    let mut blocks_per_mcu = 0;
    for c in &scan.components {
        let index = usize::from(c.component);
        if index >= geometry.components
            || c.dc > 3
            || c.ac > 3
            || std::mem::replace(&mut seen[index], true)
        {
            return Err(Error::Invalid("scan components"));
        }
        let sampling = geometry.planes[index].sampling;
        blocks_per_mcu += sampling[0] * sampling[1];
    }
    let sequential = scan.start == 0 && scan.end == 63 && scan.low == 0 && scan.high == 0;
    if sequential && progressive {
        return Err(Error::Invalid("sequential scan in progressive frame"));
    }
    if !sequential
        && (!progressive
            || scan.start > scan.end
            || scan.end > 63
            || scan.high > 13
            || scan.low > 13
            || (scan.high != 0 && scan.high != scan.low + 1)
            || (scan.start == 0 && scan.end != 0)
            || (scan.start != 0 && scan.components.len() != 1))
    {
        return Err(Error::Unsupported(
            "scan spectral or refinement declaration",
        ));
    }
    let extent = if scan.components.len() > 1 {
        geometry.mcus
    } else {
        blocks_per_mcu = 1;
        geometry.planes[usize::from(scan.components[0].component)].real_blocks
    };
    let count = u64::from(extent[0]) * u64::from(extent[1]) * u64::from(blocks_per_mcu);
    let count = u32::try_from(count).map_err(|_| Error::Invalid("scan geometry"))?;
    for (i, &block) in scan.resets.iter().enumerate() {
        if block >= count || (i != 0 && block <= scan.resets[i - 1]) {
            return Err(Error::Invalid("scan block metadata"));
        }
    }
    for (i, &(block, runs)) in scan.extra_zeros.iter().enumerate() {
        if block >= count || runs == 0 || runs > 4 || (i != 0 && block <= scan.extra_zeros[i - 1].0)
        {
            return Err(Error::Invalid("scan block metadata"));
        }
        if scan.end == 0 || scan.high != 0 {
            return Err(Error::Unsupported(
                "scan spectral or refinement declaration",
            ));
        }
    }
    Ok((extent, count))
}

fn update_progression(scan: &Scan, progression: &mut [[u32; 64]; 3]) -> Result<()> {
    let mask = if scan.high == 0 {
        u32::from(u16::MAX) & (u32::MAX << scan.low)
    } else {
        1 << scan.low
    };
    let lower = (1 << scan.low) - 1;
    for c in &scan.components {
        let plane = &mut progression[usize::from(c.component)];
        if scan.start != 0 && plane[0] == 0 {
            return Err(Error::Invalid("AC scan before DC"));
        }
        for k in scan.start..=scan.end {
            let previous = &mut plane[usize::from(k)];
            if *previous & (mask | lower) != 0
                || (scan.high != 0 && *previous & (1 << scan.high) == 0)
            {
                return Err(Error::Invalid("scan progression"));
            }
            *previous |= mask;
        }
    }
    Ok(())
}

fn walk(
    header: &Header,
    geometry: &Geometry,
    mut visit: impl FnMut(&Scan, [u32; 2], u32, u16, &[u32; 2048]) -> Result<()>,
) -> Result<[[u32; 64]; 3]> {
    if header.components().len() != geometry.components {
        return Err(Error::Invalid("component binding"));
    }
    let mut ids = [false; 256];
    for c in header.components() {
        if std::mem::replace(&mut ids[usize::from(c.id)], true) {
            return Err(Error::Invalid("component binding"));
        }
    }
    let (mut frame, mut scan_index, mut interval) = (None, 0, 0);
    let mut progression = [[0u32; 64]; 3];
    let mut tables = [0u32; 2048];
    let mut table_index = 0;
    for (index, &marker) in header.markers().iter().enumerate() {
        match marker {
            0xc0..=0xc2 => {
                if frame.replace(marker == 0xc2).is_some() {
                    return Err(Error::Invalid("marker order"));
                }
            }
            0xda => {
                let progressive = frame.ok_or(Error::Invalid("marker order"))?;
                let scan = header
                    .scans()
                    .get(scan_index)
                    .ok_or(Error::Invalid("marker order"))?;
                let (extent, count) = scan_extent(scan, geometry, progressive)?;
                update_progression(scan, &mut progression)?;
                visit(scan, extent, count, interval, &tables)?;
                scan_index += 1;
            }
            0xdd => interval = header.restart_interval(),
            0xd9 => {
                if index + 1 != header.markers().len() || scan_index == 0 {
                    return Err(Error::Invalid("marker order"));
                }
            }
            0xc4 => loop {
                let h = header
                    .huffman_tables()
                    .get(table_index)
                    .ok_or(Error::Invalid("DHT group"))?;
                table_index += 1;
                if h.values.is_empty() {
                    break;
                }
                let base = usize::from(h.index) * 256 + usize::from(h.ac) * 1024;
                tables[base..base + 256].fill(0);
                let (mut at, mut code) = (0, 0u32);
                for bits in 1..=16 {
                    for _ in 0..h.counts[bits] {
                        let &value = h.values.get(at).ok_or(Error::Invalid("Huffman counts"))?;
                        if code >= 1 << bits {
                            return Err(Error::Invalid("Huffman overflow"));
                        }
                        if value != 256 {
                            if value > 255 || code == (1 << bits) - 1 {
                                return Err(Error::Invalid("Huffman symbol"));
                            }
                            tables[base + usize::from(value)] = ((bits as u32) << 16) | code;
                        }
                        at += 1;
                        code += 1;
                    }
                    code <<= 1;
                }
                if at != h.values.len() {
                    return Err(Error::Invalid("Huffman values"));
                }
                if h.last {
                    break;
                }
            },
            0xdb | 0xe0..=0xef | 0xfe | 0xff => {}

            _ => return Err(Error::Unsupported("JPEG marker")),
        }
    }
    if header.markers().last() != Some(&0xd9) || scan_index != header.scans().len() {
        return Err(Error::Invalid("marker order"));
    }
    if progression[..geometry.components]
        .iter()
        .any(|plane| plane[0] == 0)
    {
        return Err(Error::Invalid("missing component DC scan"));
    }
    Ok(progression)
}

pub(super) fn build(
    header: &Header,
    layout: &JpegCoefficientLayout,
    limits: JpegReconstructionLimits,
) -> Result<Plan> {
    let planes = layout.planes();
    let first = planes
        .first()
        .ok_or(Error::Invalid("empty component layout"))?;
    let mcus = [
        first.blocks_per_row / first.sampling[0],
        first.block_rows / first.sampling[1],
    ];
    let geometry = Geometry {
        planes,
        components: planes.len(),
        mcus,
    };

    let mut tasks = 0u64;
    let mut owned_bytes = (header.scans().len() as u64)
        .checked_mul(size_of::<PlannedScan>() as u64 + 2048 * 4)
        .ok_or(Error::Invalid("scan geometry"))?;
    check("host scan plan bytes", owned_bytes, limits.max_plan_bytes)?;
    // Validate every scan and charge all simultaneous allocations before constructing tasks.
    let masks = walk(header, &geometry, |_, _, count, _, _| {
        tasks = tasks
            .checked_add(u64::from(count))
            .ok_or(Error::Invalid("scan geometry"))?;
        check(
            "scan tasks",
            tasks,
            limits.max_tasks.min(u64::from(u32::MAX)),
        )?;
        owned_bytes = owned_bytes
            .checked_add(u64::from(count) * size_of::<[u32; 12]>() as u64)
            .ok_or(Error::Invalid("scan geometry"))?;
        check("host scan plan bytes", owned_bytes, limits.max_plan_bytes)
    })?;
    let mut scans = allocate(header.scans().len())?;
    walk(
        header,
        &geometry,
        |scan, extent, count, interval, tables| {
            let mut output: Vec<[u32; 12]> = allocate(count as usize)?;
            let mut previous = [u32::MAX; 3];
            let (mut segment, mut restarts, mut reset_index, mut extra_index) = (0, 0u32, 0, 0);
            for my in 0..extent[1] {
                for mx in 0..extent[0] {
                    let mcu = my * extent[0] + mx;
                    if interval != 0 && mcu != 0 && mcu % u32::from(interval) == 0 {
                        output.last_mut().ok_or(Error::Invalid("scan geometry"))?[5] =
                            0xd0 + (restarts & 7);
                        restarts += 1;
                        segment = output.len() as u32;
                        previous.fill(u32::MAX);
                    }
                    for c in &scan.components {
                        let index = usize::from(c.component);
                        let plane = geometry.planes[index];
                        let [nx, ny] = if scan.components.len() > 1 {
                            plane.sampling
                        } else {
                            [1, 1]
                        };
                        for iy in 0..ny {
                            for ix in 0..nx {
                                let ordinal = output.len() as u32;
                                let reset = scan.resets.get(reset_index) == Some(&ordinal);
                                if reset {
                                    reset_index += 1;
                                }
                                let extra = match scan.extra_zeros.get(extra_index) {
                                    Some(&(block, runs)) if block == ordinal => {
                                        extra_index += 1;
                                        u32::from(runs)
                                    }
                                    _ => 0,
                                };
                                let coefficient = plane.coefficient_word_offset
                                    + ((my * ny + iy) * plane.blocks_per_row + mx * nx + ix) * 64;
                                output.push([
                                    coefficient,
                                    previous[index],
                                    u32::from(c.dc) * 256,
                                    1024 + u32::from(c.ac) * 256,
                                    extra,
                                    0,
                                    segment,
                                    restarts,
                                    u32::from(reset),
                                    0,
                                    0,
                                    0,
                                ]);
                                previous[index] = coefficient;
                            }
                        }
                    }
                }
            }
            if output.len() != count as usize
                || reset_index != scan.resets.len()
                || extra_index != scan.extra_zeros.len()
            {
                return Err(Error::Invalid("scan block metadata"));
            }
            let parameters = u32::from_le_bytes([scan.start, scan.end, scan.high, scan.low]);
            scans.push(PlannedScan {
                parameters,
                tables: {
                    let mut copy = allocate(2048)?;
                    copy.extend_from_slice(tables);
                    copy
                },
                tasks: output,
            });
            Ok(())
        },
    )?;
    Ok(Plan {
        scans,
        masks,
        owned_bytes,
    })
}

#[cfg(test)]
mod tests;
