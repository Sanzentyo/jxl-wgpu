use super::validation::{validate_huffman_counts, validate_huffman_use, validate_quant_groups};
use super::{
    AppMarker as App, AppMarkerKind as AppKind, HuffmanTable as Huffman,
    JpegComponent as Component, JpegReconstructionError as Error,
    JpegReconstructionLimits as Limits, JpegReconstructionMetadata as Header,
    JpegReconstructionResource as Resource, JpegScan as Scan, QuantizationTable as Quant,
    ScanComponent, check,
};
use crate::{BitReader, metadata::MetadataLimits};
use std::{mem::size_of, ops::Range};
type Result<T> = std::result::Result<T, Error>;

struct Reader<'a> {
    bits: BitReader<'a>,
    limits: Limits,
    owned: u64,
    entries: usize,
}
impl<'a> Reader<'a> {
    fn bits(&mut self, n: u8) -> Result<u32> {
        Ok(self.bits.read_bits(n)? as u32)
    }
    fn boolean(&mut self) -> Result<bool> {
        Ok(self.bits(1)? != 0)
    }
    fn u32(&mut self, choices: [(u8, u32); 4]) -> Result<u32> {
        let (width, base) = choices[self.bits(2)? as usize];
        base.checked_add(self.bits(width)?)
            .ok_or(Error::Invalid("u32 overflow"))
    }
    fn vec<T>(&mut self, count: usize) -> Result<Vec<T>> {
        self.entries = self
            .entries
            .checked_add(count)
            .ok_or(Error::Invalid("entry count overflow"))?;
        check(
            Resource::Entries,
            self.entries as u64,
            self.limits.max_entries as u64,
        )?;
        self.charge(
            (count as u64)
                .checked_mul(size_of::<T>() as u64)
                .ok_or(Error::Invalid("allocation size overflow"))?,
        )?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| Error::Allocation)?;
        Ok(values)
    }
    fn charge(&mut self, bytes: u64) -> Result<()> {
        self.owned = self
            .owned
            .checked_add(bytes)
            .ok_or(Error::Invalid("owned size overflow"))?;
        check(
            Resource::OwnedBytes,
            self.owned,
            self.limits.max_owned_bytes,
        )
    }
}
fn body_range(cursor: &mut usize, length: u32, limit: u64) -> Result<Range<usize>> {
    let start = *cursor;
    *cursor = cursor
        .checked_add(length as usize)
        .ok_or(Error::Invalid("body size overflow"))?;
    check(Resource::DecodedBodyBytes, *cursor as u64, limit)?;
    Ok(start..*cursor)
}
const SMALL_COUNT: [(u8, u32); 4] = [(0, 1), (0, 2), (0, 3), (0, 4)];
const LIST_COUNT: [(u8, u32); 4] = [(0, 0), (2, 1), (4, 4), (16, 20)];
const BLOCK_DELTA: [(u8, u32); 4] = [(0, 0), (3, 1), (5, 9), (28, 41)];

pub(super) fn parse(input: &[u8], limits: Limits) -> Result<Header> {
    check(
        Resource::EncodedBytes,
        input.len() as u64,
        limits.max_encoded_bytes,
    )?;
    let mut r = Reader {
        bits: BitReader::new(input),
        limits,
        owned: 0,
        entries: 0,
    };
    let gray_hint = r.boolean()?;
    // Count marker grammar before reserving exactly its storage. This never examines coefficients.
    let mut marker_count = 0;
    let (mut app_count, mut com_count, mut scan_count, mut inter_count, mut has_dri) =
        (0, 0, 0, 0, false);
    loop {
        let marker = r.bits(6)? as u8 + 0xc0;
        marker_count += 1;
        check(
            Resource::Markers,
            marker_count as u64,
            limits.max_markers.min(16384) as u64,
        )?;
        match marker {
            0xe0..=0xef => app_count += 1,
            0xfe => com_count += 1,
            0xda => scan_count += 1,
            0xff => inter_count += 1,
            0xdd => has_dri = true,
            _ => {}
        }
        if marker == 0xd9 {
            break;
        }
    }
    if scan_count == 0 {
        return Err(Error::Invalid("no scan"));
    }
    let mut markers = r.vec::<u8>(marker_count)?;
    let mut marker_reader = BitReader::new(input);
    marker_reader.read_bits(1)?;
    for _ in 0..marker_count {
        markers.push(marker_reader.read_bits(6)? as u8 + 0xc0);
    }
    let mut body_size = 0;
    let mut apps = r.vec::<App>(app_count)?;
    for _ in 0..app_count {
        let kind = match r.u32([(0, 0), (0, 1), (1, 2), (2, 4)])? {
            0 => AppKind::Unknown,
            1 => AppKind::Icc,
            2 => AppKind::Exif,
            3 => AppKind::Xmp,
            _ => return Err(Error::Invalid("APP kind")),
        };
        let size = r.bits(16)? + 1;
        let minimum = match kind {
            AppKind::Unknown => 3,
            AppKind::Icc => 17,
            AppKind::Exif => 9,
            AppKind::Xmp => 32,
        };
        if size < minimum {
            return Err(Error::Invalid("APP length"));
        }
        let body = if kind == AppKind::Unknown {
            Some(body_range(
                &mut body_size,
                size,
                limits.max_decoded_body_bytes,
            )?)
        } else {
            None
        };
        apps.push(App { kind, size, body });
    }
    let mut comments = r.vec::<Range<usize>>(com_count)?;
    for _ in 0..com_count {
        let size = r.bits(16)? + 1;
        if size < 3 {
            return Err(Error::Invalid("COM length"));
        }
        comments.push(body_range(
            &mut body_size,
            size,
            limits.max_decoded_body_bytes,
        )?);
    }
    let count = r.u32(SMALL_COUNT)? as usize;
    if count == 4 {
        return Err(Error::Invalid("quantization table count"));
    }
    let mut quant = r.vec::<Quant>(count)?;
    for _ in 0..count {
        quant.push(Quant {
            precision: r.bits(1)? as u8,
            index: r.bits(2)? as u8,
            last: r.boolean()?,
        });
    }
    let kind = r.bits(2)?;
    let count = match kind {
        0 => 1,
        3 => r.u32(SMALL_COUNT)?,
        _ => 3,
    } as usize;
    if count != 1 && count != 3 {
        return Err(Error::Invalid("component count"));
    }
    let mut components = r.vec::<Component>(count)?;
    for i in 0..count {
        components.push(Component {
            id: match kind {
                3 => r.bits(8)? as u8,
                2 => b"RGB"[i],
                _ => i as u8 + 1,
            },
            quant: 0,
        });
    }
    let mut used = 0;
    for c in &mut components {
        c.quant = r.bits(2)? as u8;
        if c.quant as usize >= quant.len() {
            return Err(Error::Invalid("component quantization selector"));
        }
        used |= 1 << c.quant;
    }
    if used & 1 == 0 {
        return Err(Error::Invalid("unused first quantization table"));
    }
    let count = r.u32([(0, 4), (3, 2), (4, 10), (6, 26)])? as usize;
    let mut huffman = r.vec::<Huffman>(count)?;
    for _ in 0..count {
        let ac = r.boolean()?;
        let index = r.bits(2)? as u8;
        let last = r.boolean()?;
        let mut counts = [0u16; 17];
        for v in &mut counts {
            *v = r.u32([(0, 0), (0, 1), (3, 2), (8, 0)])? as u16;
        }
        let count = counts.into_iter().map(usize::from).sum::<usize>();
        if count > 257 {
            return Err(Error::Invalid("Huffman symbol count"));
        }
        let mut values = r.vec::<u16>(count)?;
        let mut seen = [false; 257];
        for _ in 0..count {
            let value = r.u32([(2, 0), (2, 4), (4, 8), (8, 1)])? as u16;
            if std::mem::replace(&mut seen[value as usize], true) {
                return Err(Error::Invalid("duplicate Huffman symbol"));
            }
            if !ac && value >= 12 && value != 256 {
                return Err(Error::Invalid("DC Huffman symbol"));
            }
            values.push(value);
        }
        if count != 0 && values.last() != Some(&256) {
            return Err(Error::Invalid("Huffman terminal symbol"));
        }
        validate_huffman_counts(&counts)?;
        huffman.push(Huffman {
            ac,
            index,
            last,
            counts,
            values,
        });
    }
    let mut scans = r.vec::<Scan>(scan_count)?;
    for _ in 0..scan_count {
        let count = r.u32(SMALL_COUNT)? as usize;
        if count >= 4 {
            return Err(Error::Invalid("scan component count"));
        }
        let start = r.bits(6)? as u8;
        let end = r.bits(6)? as u8;
        let low = r.bits(4)? as u8;
        let high = r.bits(4)? as u8;
        let mut scan_components = r.vec::<ScanComponent>(count)?;
        for _ in 0..count {
            let component = r.bits(2)? as u8;
            if component as usize >= components.len() {
                return Err(Error::Invalid("scan component selector"));
            }
            scan_components.push(ScanComponent {
                component,
                ac: r.bits(2)? as u8,
                dc: r.bits(2)? as u8,
            });
        }
        let last_pass = r.u32([(0, 0), (0, 1), (0, 2), (3, 3)])? as u8;
        scans.push(Scan {
            start,
            end,
            low,
            high,
            components: scan_components,
            last_pass,
            resets: Vec::new(),
            extra_zeros: Vec::new(),
        });
    }
    let restart_interval = if has_dri { r.bits(16)? as u16 } else { 0 };
    for scan in &mut scans {
        let count = r.u32(LIST_COUNT)? as usize;
        scan.resets = r.vec::<u32>(count)?;
        let mut next = 0u32;
        for _ in 0..count {
            let block = next
                .checked_add(r.u32(BLOCK_DELTA)?)
                .ok_or(Error::Invalid("reset block overflow"))?;
            if block >= 3 << 26 {
                return Err(Error::Invalid("reset block index"));
            }
            scan.resets.push(block);
            next = block + 1;
        }
        let count = r.u32(LIST_COUNT)? as usize;
        scan.extra_zeros = r.vec::<(u32, u8)>(count)?;
        next = 0;
        for _ in 0..count {
            let runs = r.u32([(0, 1), (2, 2), (4, 5), (8, 20)])?;
            if runs > 4 {
                return Err(Error::Invalid("extra zero run count"));
            }
            let block = next
                .checked_add(r.u32(BLOCK_DELTA)?)
                .ok_or(Error::Invalid("extra zero block overflow"))?;
            // Preserve native's inclusive metadata endpoint pending actual-image grid validation.
            if block > 3 << 26 {
                return Err(Error::Invalid("extra zero block index"));
            }
            scan.extra_zeros.push((block, runs as u8));
            next = block + 1;
        }
    }
    let mut intermarker = r.vec::<Range<usize>>(inter_count)?;
    for _ in 0..inter_count {
        intermarker.push(body_range(
            &mut body_size,
            r.bits(16)?,
            limits.max_decoded_body_bytes,
        )?);
    }
    let tail_len = r.u32([(0, 0), (8, 1), (16, 257), (22, 65793)])?;
    let tail = body_range(&mut body_size, tail_len, limits.max_decoded_body_bytes)?;
    let has_zero_padding = r.boolean()?;
    let padding_bits = if has_zero_padding { r.bits(24)? } else { 0 };
    let mut padding = r.vec::<u8>(padding_bits.div_ceil(8) as usize)?;
    for byte in 0..padding_bits.div_ceil(8) {
        padding.push(r.bits((padding_bits - byte * 8).min(8) as u8)? as u8);
    }
    validate_quant_groups(&markers, &quant)?;
    validate_huffman_use(&markers, &huffman, &scans)?;
    let align_bits = ((8 - r.bits.bit_offset() % 8) % 8) as u8;
    if r.bits(align_bits)? != 0 {
        return Err(Error::Invalid("nonzero header padding"));
    }
    let header_bytes = usize::try_from(r.bits.bit_offset() / 8)
        .map_err(|_| Error::Invalid("header offset overflow"))?;
    r.charge(body_size as u64)?;
    // Share the strict metadata-body decoder; jbrd remains forbidden inside brob.
    let stream = input
        .get(header_bytes..)
        .ok_or(Error::Invalid("body offset"))?;
    let body = crate::metadata::decompress_body(
        stream,
        MetadataLimits {
            max_encoded_box_bytes: limits.max_encoded_bytes,
            max_decoded_box_bytes: body_size as u64,
            max_brotli_window_bits: limits.max_brotli_window_bits,
            max_expansion_ratio: limits.max_expansion_ratio,
            ..Default::default()
        },
    )?;
    if body.len() != body_size {
        return Err(Error::Invalid("decompressed metadata length"));
    }
    for range in apps
        .iter()
        .filter_map(|app| app.body.as_ref())
        .chain(&comments)
    {
        let record = &body[range.clone()];
        if usize::from(u16::from_be_bytes([record[1], record[2]])) + 1 != record.len() {
            return Err(Error::Invalid("APP/COM encoded length"));
        }
    }
    Ok(Header {
        gray_hint,
        markers,
        apps,
        comments,
        quant,
        components,
        huffman,
        scans,
        restart_interval,
        intermarker,
        tail,
        padding,
        padding_bits,
        has_zero_padding,
        body,
        header_bytes,
        logical_owned_bytes: r.owned,
    })
}
