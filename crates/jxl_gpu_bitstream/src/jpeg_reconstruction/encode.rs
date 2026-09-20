use super::{
    AppMarkerKind as AppKind, EncodedJpegReconstruction as Emitted,
    JpegReconstructionEncodeOptions as Options, JpegReconstructionError as Error,
    JpegReconstructionMetadata as Header, JpegReconstructionResource as Resource, check,
};
use crate::metadata::{MetadataError, MetadataLimits, MetadataResource};
type Result<T> = std::result::Result<T, Error>;

struct Bits {
    storage: Option<Vec<u8>>,
    bit_offset: usize,
}
impl Bits {
    fn counting() -> Self {
        Self {
            storage: None,
            bit_offset: 0,
        }
    }
    fn allocated(bytes: usize) -> Result<Self> {
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(bytes)
            .map_err(|_| Error::Allocation)?;
        storage.resize(bytes, 0);
        Ok(Self {
            storage: Some(storage),
            bit_offset: 0,
        })
    }
    fn bit_len(&self) -> usize {
        self.bit_offset
    }
    fn write_bits(&mut self, value: u64, count: u8) -> Result<()> {
        if count > 32 || value >= (1u64 << count) {
            return Err(Error::Invalid("metadata bit value"));
        }
        let end = self
            .bit_offset
            .checked_add(usize::from(count))
            .ok_or(Error::Invalid("header bit count overflow"))?;
        if let Some(storage) = &mut self.storage {
            if end.div_ceil(8) > storage.len() {
                return Err(Error::Invalid("header count changed"));
            }
            for bit in 0..count {
                let at = self.bit_offset + usize::from(bit);
                storage[at / 8] |= (((value >> bit) & 1) as u8) << (at % 8);
            }
        }
        self.bit_offset = end;
        Ok(())
    }
    fn align_to_byte(&mut self) -> Result<()> {
        self.write_bits(0, ((8 - (self.bit_offset & 7)) & 7) as u8)
    }
    fn into_bytes(self) -> Result<Vec<u8>> {
        self.storage.ok_or(Error::Invalid("missing header output"))
    }
}
struct Writer {
    writer: Bits,
    limit: u64,
}
impl Writer {
    fn bits(&mut self, value: u32, width: u8) -> Result<()> {
        let bytes = (self.writer.bit_len() as u64)
            .checked_add(u64::from(width))
            .ok_or(Error::Invalid("header bit count overflow"))?
            .div_ceil(8);
        if bytes > self.limit {
            return Err(Error::Limit {
                resource: Resource::EncodedBytes,
                required: bytes,
                limit: self.limit,
            });
        }
        self.writer.write_bits(u64::from(value), width)
    }
    fn boolean(&mut self, value: bool) -> Result<()> {
        self.bits(u32::from(value), 1)
    }
    fn u32(&mut self, value: u32, choices: [(u8, u32); 4]) -> Result<()> {
        let (index, (width, base)) = choices
            .into_iter()
            .enumerate()
            .filter(|(_, (width, base))| value >= *base && u64::from(value - *base) < 1u64 << width)
            .min_by_key(|(index, (width, _))| (*width, *index))
            .ok_or(Error::Invalid("unrepresentable metadata value"))?;
        self.bits(index as u32, 2)?;
        self.bits(value - base, width)
    }
}
const SMALL: [(u8, u32); 4] = [(0, 1), (0, 2), (0, 3), (0, 4)];
const LIST: [(u8, u32); 4] = [(0, 0), (2, 1), (4, 4), (16, 20)];
const DELTA: [(u8, u32); 4] = [(0, 0), (3, 1), (5, 9), (28, 41)];
fn length(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| Error::Invalid("metadata length overflow"))
}
fn write_header(header: &Header, w: &mut Writer) -> Result<()> {
    w.boolean(header.gray_hint)?;
    for &marker in &header.markers {
        w.bits(
            u32::from(marker.checked_sub(0xc0).ok_or(Error::Invalid("marker"))?),
            6,
        )?;
    }
    for app in &header.apps {
        w.u32(
            match app.kind {
                AppKind::Unknown => 0,
                AppKind::Icc => 1,
                AppKind::Exif => 2,
                AppKind::Xmp => 3,
            },
            [(0, 0), (0, 1), (1, 2), (2, 4)],
        )?;
        w.bits(
            app.size
                .checked_sub(1)
                .ok_or(Error::Invalid("APP length"))?,
            16,
        )?;
    }
    for range in &header.comments {
        w.bits(
            length(range.len())?
                .checked_sub(1)
                .ok_or(Error::Invalid("COM length"))?,
            16,
        )?;
    }
    w.u32(length(header.quant.len())?, SMALL)?;
    for q in &header.quant {
        w.bits(u32::from(q.precision), 1)?;
        w.bits(u32::from(q.index), 2)?;
        w.boolean(q.last)?;
    }
    let ids: [u8; 3] = std::array::from_fn(|i| header.components.get(i).map_or(0, |c| c.id));
    let kind = match (header.components.len(), ids) {
        (1, [1, ..]) => 0,
        (3, [1, 2, 3]) => 1,
        (3, [b'R', b'G', b'B']) => 2,
        _ => 3,
    };
    w.bits(kind, 2)?;
    if kind == 3 {
        w.u32(length(header.components.len())?, SMALL)?;
        for id in header.components.iter().map(|c| c.id) {
            w.bits(u32::from(id), 8)?;
        }
    }
    for c in &header.components {
        w.bits(u32::from(c.quant), 2)?;
    }
    w.u32(
        length(header.huffman.len())?,
        [(0, 4), (3, 2), (4, 10), (6, 26)],
    )?;
    for h in &header.huffman {
        w.boolean(h.ac)?;
        w.bits(u32::from(h.index), 2)?;
        w.boolean(h.last)?;
        for count in h.counts {
            w.u32(u32::from(count), [(0, 0), (0, 1), (3, 2), (8, 0)])?;
        }
        for &value in &h.values {
            w.u32(u32::from(value), [(2, 0), (2, 4), (4, 8), (8, 1)])?;
        }
    }
    for s in &header.scans {
        w.u32(length(s.components.len())?, SMALL)?;
        w.bits(u32::from(s.start), 6)?;
        w.bits(u32::from(s.end), 6)?;
        w.bits(u32::from(s.low), 4)?;
        w.bits(u32::from(s.high), 4)?;
        for c in &s.components {
            w.bits(u32::from(c.component), 2)?;
            w.bits(u32::from(c.ac), 2)?;
            w.bits(u32::from(c.dc), 2)?;
        }
        w.u32(u32::from(s.last_pass), [(0, 0), (0, 1), (0, 2), (3, 3)])?;
    }
    if header.markers.contains(&0xdd) {
        w.bits(u32::from(header.restart_interval), 16)?;
    }
    for s in &header.scans {
        w.u32(length(s.resets.len())?, LIST)?;
        let mut next = 0;
        for &block in &s.resets {
            w.u32(
                block
                    .checked_sub(next)
                    .ok_or(Error::Invalid("reset order"))?,
                DELTA,
            )?;
            next = block + 1;
        }
        w.u32(length(s.extra_zeros.len())?, LIST)?;
        next = 0;
        for &(block, runs) in &s.extra_zeros {
            w.u32(u32::from(runs), [(0, 1), (2, 2), (4, 5), (8, 20)])?;
            w.u32(
                block
                    .checked_sub(next)
                    .ok_or(Error::Invalid("extra zero order"))?,
                DELTA,
            )?;
            next = block + 1;
        }
    }
    for range in &header.intermarker {
        w.bits(length(range.len())?, 16)?;
    }
    w.u32(
        length(header.tail.len())?,
        [(0, 0), (8, 1), (16, 257), (22, 65793)],
    )?;
    w.boolean(header.has_zero_padding)?;
    if header.has_zero_padding {
        w.bits(header.padding_bits, 24)?;
        for i in 0..header.padding_bits {
            w.bits(
                u32::from(
                    (*header
                        .padding
                        .get(i as usize / 8)
                        .ok_or(Error::Invalid("padding bit storage"))?
                        >> (i % 8))
                        & 1,
                ),
                1,
            )?;
        }
    }
    w.writer.align_to_byte()?;
    Ok(())
}
pub(super) fn encode(header: &Header, limits: Options) -> Result<Emitted> {
    let mut counter = Writer {
        writer: Bits::counting(),
        limit: limits.max_encoded_bytes,
    };
    write_header(header, &mut counter)?;
    let header_bytes = counter.writer.bit_len() / 8;
    let fixed = header
        .logical_owned_bytes
        .checked_add(header_bytes as u64)
        .ok_or(Error::Invalid("metadata ownership overflow"))?;
    check(Resource::OwnedBytes, fixed, limits.max_owned_bytes)?;
    let encoded_allowance = limits.max_encoded_bytes - header_bytes as u64;
    let owned_allowance = (limits.max_owned_bytes - fixed) / 2;
    let allowance = encoded_allowance.min(owned_allowance);
    let compressed = crate::metadata::compress_with_prefix(
        &[],
        &header.body,
        limits.brotli,
        MetadataLimits {
            max_encoded_box_bytes: allowance,
            max_decoded_box_bytes: header.body.len() as u64,
            max_brotli_window_bits: limits.max_brotli_window_bits,
            max_expansion_ratio: limits.max_expansion_ratio,
            ..Default::default()
        },
    )
    .map_err(|error| match error {
        MetadataError::Limit {
            resource: MetadataResource::EncodedBoxBytes,
            bytes,
            ..
        } => {
            if owned_allowance < encoded_allowance {
                Error::Limit {
                    resource: Resource::OwnedBytes,
                    required: fixed.saturating_add(bytes.saturating_mul(2)),
                    limit: limits.max_owned_bytes,
                }
            } else {
                Error::Limit {
                    resource: Resource::EncodedBytes,
                    required: (header_bytes as u64).saturating_add(bytes),
                    limit: limits.max_encoded_bytes,
                }
            }
        }
        other => Error::Metadata(other),
    })?;
    let total = header_bytes
        .checked_add(compressed.len())
        .ok_or(Error::Invalid("encoded metadata size overflow"))?;
    let logical_peak_owned_bytes = fixed
        .checked_add(
            (compressed.len() as u64)
                .checked_mul(2)
                .ok_or(Error::Invalid("metadata ownership overflow"))?,
        )
        .ok_or(Error::Invalid("metadata ownership overflow"))?;
    check(
        Resource::OwnedBytes,
        logical_peak_owned_bytes,
        limits.max_owned_bytes,
    )?;
    check(
        Resource::EncodedBytes,
        total as u64,
        limits.max_encoded_bytes,
    )?;
    let mut output = Writer {
        writer: Bits::allocated(total)?,
        limit: header_bytes as u64,
    };
    write_header(header, &mut output)?;
    if output.writer.bit_len() != counter.writer.bit_len() {
        return Err(Error::Invalid("header count changed"));
    }
    let mut bytes = output.writer.into_bytes()?;
    bytes[header_bytes..].copy_from_slice(&compressed);
    Ok(Emitted {
        bytes,
        header_bytes,
        compressed_body_bytes: compressed.len(),
        logical_peak_owned_bytes,
    })
}
