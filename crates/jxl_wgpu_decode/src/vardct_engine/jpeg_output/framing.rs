//! Metadata-only JPEG framing. Quantizers are placeholders patched from the GPU coefficient lease.

use super::super::jpeg::JpegCoefficientLayout;
use super::{Error, Result, add, allocate, check};
use jxl_gpu_bitstream::jpeg_reconstruction::{AppMarkerKind, JpegReconstructionMetadata as Header};

#[derive(Default)]
pub(super) struct Extras<'a> {
    pub icc: &'a [u8],
    pub exif: &'a [u8],
    pub xmp: &'a [u8],
}

#[derive(Debug)]
pub(super) struct Plan {
    pub bytes: Vec<u8>,
    /// Offset after each SOS header in the template, where its GPU scan must be inserted.
    pub scan_offsets: Vec<u32>,
    /// Template byte offset, source quantizer word offset, internal-channel mask, precision.
    pub quantizers: Vec<[u32; 4]>,
    pub owned_bytes: u64,
}

struct Sink<'a> {
    output: Option<&'a mut Plan>,
    length: u64,
    limit: u64,
}

impl Sink<'_> {
    fn put(&mut self, bytes: &[u8]) -> Result<()> {
        self.length = add(self.length, bytes.len() as u64)?;
        check(
            "JPEG marker bytes",
            self.length,
            self.limit.min(u64::from(u32::MAX)),
        )?;
        if let Some(output) = &mut self.output {
            output.bytes.extend_from_slice(bytes);
        }
        Ok(())
    }

    fn marker(&mut self, marker: u8, payload: usize) -> Result<()> {
        let size = u16::try_from(
            payload
                .checked_add(2)
                .ok_or(Error::Invalid("marker size"))?,
        )
        .map_err(|_| Error::Invalid("marker size"))?;
        self.put(&[0xff, marker])?;
        self.put(&size.to_be_bytes())
    }

    fn quantizer(&mut self, source: [u32; 2], precision: u8) -> Result<()> {
        if let Some(output) = &mut self.output {
            output.quantizers.push([
                self.length as u32,
                source[0],
                source[1],
                u32::from(precision),
            ]);
        }
        self.put(&[0u8; 128][..64 * (1 + usize::from(precision))])
    }
}

fn quantizer_sources(header: &Header, layout: &JpegCoefficientLayout) -> Result<Vec<[u32; 2]>> {
    let quant = header.quantization_tables();
    let mut result = allocate(quant.len())?;
    result.resize(quant.len(), [0u32; 2]);
    for (component, plane) in header.components().iter().zip(layout.planes()) {
        let record = result
            .get_mut(usize::from(component.quant))
            .ok_or(Error::Invalid("component quantizer"))?;
        record[1] |= 1 << (plane.quantization_word_offset / 64);
    }
    let mut active = [None; 4];
    let (mut q, mut s) = (0, 0);
    for &marker in header.markers() {
        if marker == 0xdb {
            loop {
                let table = quant.get(q).ok_or(Error::Invalid("DQT group"))?;
                active[usize::from(table.index)] = Some(q);
                q += 1;
                if table.last {
                    break;
                }
            }
        } else if marker == 0xda {
            let scan = header.scans().get(s).ok_or(Error::Invalid("SOS count"))?;
            s += 1;
            for c in &scan.components {
                let index = usize::from(c.component);
                let component = header
                    .components()
                    .get(index)
                    .ok_or(Error::Invalid("SOS component"))?;
                let selector = quant[usize::from(component.quant)].index;
                let record = active[usize::from(selector)]
                    .ok_or(Error::Invalid("quantizer used before DQT"))?;
                result[record][1] |= 1 << (layout.planes()[index].quantization_word_offset / 64);
            }
        }
    }
    let mut previous = None;
    for record in &mut result {
        let base = if record[1] == 0 {
            previous.ok_or(Error::Unsupported("unbound first quantization table"))?
        } else {
            record[1].trailing_zeros() * 64
        };
        record[0] = base;
        previous = Some(base);
    }
    Ok(result)
}

fn write(
    sink: &mut Sink<'_>,
    header: &Header,
    layout: &JpegCoefficientLayout,
    extras: &Extras<'_>,
    quantizers: &[[u32; 2]],
) -> Result<()> {
    let [width, height] = layout.extent();
    let components = header.components();
    let quant = header.quantization_tables();
    let huffman = header.huffman_tables();
    let icc_count = u8::try_from(
        header
            .app_markers()
            .iter()
            .filter(|a| a.kind == AppMarkerKind::Icc)
            .count(),
    )
    .map_err(|_| Error::Invalid("ICC chunk count"))?;
    let (mut app, mut q, mut h, mut s) = (0, 0, 0, 0);
    let (mut icc_chunk, mut icc_offset, mut exif_count, mut xmp_count) = (0u8, 0usize, 0, 0);
    let mut comments = header.comments();
    let mut intermarker = header.intermarker_data();
    sink.put(&[0xff, 0xd8])?;
    for &marker in header.markers() {
        match marker {
            0xc0..=0xc2 => {
                sink.marker(marker, 6 + 3 * components.len())?;
                sink.put(&[8])?;
                sink.put(&(height as u16).to_be_bytes())?;
                sink.put(&(width as u16).to_be_bytes())?;
                sink.put(&[components.len() as u8])?;
                for (component, plane) in components.iter().zip(layout.planes()) {
                    sink.put(&[
                        component.id,
                        ((plane.sampling[0] << 4) | plane.sampling[1]) as u8,
                        quant[usize::from(component.quant)].index,
                    ])?;
                }
            }
            0xdb => {
                let first = q;
                let mut length = 0;
                loop {
                    let table = quant.get(q).ok_or(Error::Invalid("DQT group"))?;
                    q += 1;
                    length += 1 + 64 * (1 + usize::from(table.precision));
                    if table.last {
                        break;
                    }
                }
                sink.marker(marker, length)?;
                for (table, &source) in quant[first..q].iter().zip(&quantizers[first..q]) {
                    sink.put(&[(table.precision << 4) | table.index])?;
                    sink.quantizer(source, table.precision)?;
                }
            }
            0xc4 => {
                let first = h;
                let mut length = 0;
                loop {
                    let table = huffman.get(h).ok_or(Error::Invalid("DHT group"))?;
                    h += 1;
                    if table.values.is_empty() {
                        break;
                    }
                    length += 16 + table.values.len();
                    if table.last {
                        break;
                    }
                }
                sink.marker(marker, length)?;
                for table in &huffman[first..h] {
                    if table.values.is_empty() {
                        continue;
                    }
                    sink.put(&[table.index | if table.ac { 0x10 } else { 0 }])?;
                    let longest = table
                        .counts
                        .iter()
                        .rposition(|&count| count != 0)
                        .ok_or(Error::Invalid("DHT counts"))?;
                    for bits in 1..=16 {
                        sink.put(&[
                            u8::try_from(table.counts[bits] - u16::from(bits == longest))
                                .map_err(|_| Error::Invalid("DHT count"))?,
                        ])?;
                    }
                    for &value in &table.values[..table.values.len() - 1] {
                        sink.put(&[u8::try_from(value).map_err(|_| Error::Invalid("DHT value"))?])?;
                    }
                }
            }
            0xda => {
                let scan = header.scans().get(s).ok_or(Error::Invalid("SOS count"))?;
                sink.marker(marker, 4 + 2 * scan.components.len())?;
                sink.put(&[scan.components.len() as u8])?;
                for c in &scan.components {
                    sink.put(&[components[usize::from(c.component)].id, (c.dc << 4) | c.ac])?;
                }
                sink.put(&[scan.start, scan.end, (scan.high << 4) | scan.low])?;
                if let Some(output) = &mut sink.output {
                    output.scan_offsets.push(sink.length as u32);
                }
                s += 1;
            }
            0xdd => {
                sink.marker(marker, 2)?;
                sink.put(&header.restart_interval().to_be_bytes())?;
            }
            0xe0..=0xef => {
                let record = header
                    .app_markers()
                    .get(app)
                    .ok_or(Error::Invalid("APP count"))?;
                app += 1;
                if let Some(range) = &record.body {
                    let bytes = &header.opaque_body()[range.clone()];
                    if bytes.first() != Some(&marker) {
                        return Err(Error::Invalid("APP identity"));
                    }
                    sink.put(&[0xff])?;
                    sink.put(bytes)?;
                    continue;
                }
                sink.marker(marker, record.size as usize - 3)?;
                match record.kind {
                    AppMarkerKind::Icc => {
                        if marker != 0xe2 {
                            return Err(Error::Invalid("ICC marker"));
                        }
                        icc_chunk += 1;
                        sink.put(b"ICC_PROFILE\0")?;
                        sink.put(&[icc_chunk, icc_count])?;
                        let next = icc_offset
                            .checked_add(record.size as usize - 17)
                            .ok_or(Error::Invalid("ICC offset"))?;
                        sink.put(
                            extras
                                .icc
                                .get(icc_offset..next)
                                .ok_or(Error::Invalid("ICC length"))?,
                        )?;
                        icc_offset = next;
                    }
                    AppMarkerKind::Exif => {
                        if marker != 0xe1
                            || exif_count != 0
                            || extras.exif.len() < 4
                            || extras.exif.len() - 4 != record.size as usize - 9
                        {
                            return Err(Error::Invalid("Exif binding"));
                        }
                        exif_count += 1;
                        sink.put(b"Exif\0\0")?;
                        sink.put(&extras.exif[4..])?;
                    }
                    AppMarkerKind::Xmp => {
                        if marker != 0xe1
                            || xmp_count != 0
                            || extras.xmp.len() != record.size as usize - 32
                        {
                            return Err(Error::Invalid("XMP binding"));
                        }
                        xmp_count += 1;
                        sink.put(b"http://ns.adobe.com/xap/1.0/\0")?;
                        sink.put(extras.xmp)?;
                    }
                    AppMarkerKind::Unknown => return Err(Error::Invalid("missing APP body")),
                }
            }
            0xfe => {
                let bytes = comments.next().ok_or(Error::Invalid("COM count"))?;
                if bytes.first() != Some(&marker) {
                    return Err(Error::Invalid("COM identity"));
                }
                sink.put(&[0xff])?;
                sink.put(bytes)?;
            }
            0xff => sink.put(
                intermarker
                    .next()
                    .ok_or(Error::Invalid("intermarker count"))?,
            )?,
            0xd9 => {
                sink.put(&[0xff, marker])?;
                sink.put(header.tail())?;
            }
            _ => return Err(Error::Unsupported("JPEG marker")),
        }
    }
    if app != header.app_markers().len()
        || q != quant.len()
        || h != huffman.len()
        || s != header.scans().len()
        || comments.len() != 0
        || intermarker.len() != 0
        || (icc_count != 0 && icc_offset != extras.icc.len())
    {
        return Err(Error::Invalid("unconsumed framing metadata"));
    }
    Ok(())
}

pub(super) fn build(
    header: &Header,
    layout: &JpegCoefficientLayout,
    extras: &Extras<'_>,
    byte_limit: u64,
    host_limit: u64,
) -> Result<Plan> {
    let scratch_bytes = header.quantization_tables().len() as u64 * 8;
    check("framing plan bytes", scratch_bytes, host_limit)?;
    let sources = quantizer_sources(header, layout)?;
    let mut sink = Sink {
        output: None,
        length: 0,
        limit: byte_limit,
    };
    write(&mut sink, header, layout, extras, &sources)?;
    let owned_bytes = add(
        sink.length,
        add(header.scans().len() as u64 * 4, sources.len() as u64 * 16)?,
    )?;
    check(
        "framing plan bytes",
        add(owned_bytes, scratch_bytes)?,
        host_limit,
    )?;
    let mut plan = Plan {
        bytes: allocate(sink.length as usize)?,
        scan_offsets: allocate(header.scans().len())?,
        quantizers: allocate(sources.len())?,
        owned_bytes,
    };
    let mut sink = Sink {
        output: Some(&mut plan),
        length: 0,
        limit: byte_limit,
    };
    write(&mut sink, header, layout, extras, &sources)?;
    Ok(plan)
}
