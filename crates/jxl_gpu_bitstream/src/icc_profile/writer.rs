// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// libjxl 0.12.0 ICC layout and fixed-point serialization, BSD-3-Clause.
use super::{IccProfileError as Error, Result, encoding::Encoding, math::Matrix};
use crate::{ColourSpaceInventory as Space, TransferFunctionInventory as Transfer};

#[derive(Clone, Copy)]
pub(super) enum Kind {
    Description,
    Copyright,
    White,
    Adaptation,
    Cicp,
    Primary(usize),
    Curve,
    Xyb,
    Hdr,
    Reverse,
}
#[derive(Clone, Copy)]
pub(super) struct Tag {
    pub kind: Kind,
    pub size: usize,
    names: [[u8; 4]; 3],
    count: usize,
}
impl Tag {
    pub fn signatures(&self) -> &[[u8; 4]] {
        &self.names[..self.count]
    }
}
pub(super) struct Plan {
    tags: [Tag; 10],
    count: usize,
    pub tag_count: usize,
    pub bytes: usize,
}
impl Plan {
    pub fn new(e: &Encoding) -> Result<Self> {
        let mut p = Self {
            tags: [Tag {
                kind: Kind::White,
                size: 0,
                names: [[0; 4]; 3],
                count: 0,
            }; 10],
            count: 0,
            tag_count: 0,
            bytes: 132,
        };
        p.add(
            Kind::Description,
            b"desc",
            28 + 2 * e.description.as_bytes().len(),
        );
        p.add(Kind::Copyright, b"cprt", 34);
        p.add(Kind::White, b"wtpt", 20);
        if e.space != Space::Grey {
            p.add(Kind::Adaptation, b"chad", 44);
        }
        if e.cicp.is_some() {
            p.add(Kind::Cicp, b"cicp", 12);
        }
        if e.space == Space::Rgb {
            for (c, name) in [b"rXYZ", b"gXYZ", b"bXYZ"].into_iter().enumerate() {
                p.add(Kind::Primary(c), name, 20);
            }
        }
        if e.space == Space::Xyb || e.hdr {
            p.add(
                if e.hdr { Kind::Hdr } else { Kind::Xyb },
                b"A2B0",
                if e.hdr { 3771 } else { 292 },
            );
            p.add(Kind::Reverse, b"B2A0", 80);
        } else {
            let size = match e.transfer {
                Transfer::Gamma { .. } => {
                    fixed((1.0 / e.gamma) as f32)?;
                    16
                }
                Transfer::Pq | Transfer::Hlg => 140,
                _ => 32,
            };
            p.add(
                Kind::Curve,
                if e.space == Space::Grey {
                    b"kTRC"
                } else {
                    b"rTRC"
                },
                size,
            );
            if e.space == Space::Rgb {
                let tag = &mut p.tags[p.count - 1];
                tag.names[1] = *b"gTRC";
                tag.names[2] = *b"bTRC";
                tag.count = 3;
                p.tag_count += 2;
                p.bytes += 24;
            }
        }
        // Check all serialized matrix values before allocating the output.
        for value in e
            .white
            .into_iter()
            .chain(e.adaptation.into_iter().flatten())
            .chain(e.primaries.into_iter().flatten())
        {
            fixed(value)?;
        }
        Ok(p)
    }
    fn add(&mut self, kind: Kind, name: &[u8; 4], size: usize) {
        let size = size.next_multiple_of(4);
        self.tags[self.count] = Tag {
            kind,
            size,
            names: [*name, [0; 4], [0; 4]],
            count: 1,
        };
        self.count += 1;
        self.tag_count += 1;
        self.bytes += 12 + size;
    }
    pub fn tags(&self) -> &[Tag] {
        &self.tags[..self.count]
    }
}

fn fixed(value: f32) -> Result<u32> {
    if !(-32767.995..=32767.995).contains(&value) {
        return Err(Error::Invalid("ICC fixed-point value"));
    }
    Ok((value * 65536.0).round() as i32 as u32)
}

pub(super) struct Writer {
    pub bytes: Vec<u8>,
    pub cursor: usize,
}
impl Writer {
    pub fn new(size: usize) -> Result<Self> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| Error::Allocation { bytes: size as u64 })?;
        bytes.resize(size, 0);
        Ok(Self { bytes, cursor: 0 })
    }
    pub fn bytes_at(&mut self, at: usize, value: &[u8]) {
        self.bytes[at..at + value.len()].copy_from_slice(value);
    }
    pub fn u32_at(&mut self, at: usize, value: u32) {
        self.bytes_at(at, &value.to_be_bytes());
    }
    pub fn append(&mut self, value: &[u8]) {
        self.bytes_at(self.cursor, value);
        self.cursor += value.len();
    }
    pub fn u32(&mut self, value: u32) {
        self.append(&value.to_be_bytes());
    }
    pub fn u16(&mut self, value: u16) {
        self.append(&value.to_be_bytes());
    }
    pub fn u8(&mut self, value: u8) {
        self.append(&[value]);
    }
    pub fn fixed(&mut self, value: f32) -> Result<()> {
        self.u32(fixed(value)?);
        Ok(())
    }
    pub fn header(&mut self, e: &Encoding) {
        self.u32_at(0, self.bytes.len() as u32);
        self.bytes_at(4, b"jxl ");
        self.u32_at(8, 0x04400000);
        self.bytes_at(
            12,
            if e.space == Space::Xyb {
                b"scnr"
            } else {
                b"mntr"
            },
        );
        self.bytes_at(
            16,
            if e.space == Space::Grey {
                b"GRAY"
            } else {
                b"RGB "
            },
        );
        self.bytes_at(20, if e.hdr { b"Lab " } else { b"XYZ " });
        self.bytes_at(24, &[0x07, 0xe3, 0, 12, 0, 1]);
        self.bytes_at(36, b"acsp");
        self.bytes_at(40, b"APPL");
        self.u32_at(64, e.intent as u32);
        for (c, value) in [0xf6d6, 0x10000, 0xd32d].into_iter().enumerate() {
            self.u32_at(68 + c * 4, value);
        }
        self.bytes_at(80, b"jxl ");
    }
    pub fn mluc(&mut self, text: &[u8]) {
        self.append(b"mluc");
        self.u32(0);
        self.u32(1);
        self.u32(12);
        self.append(b"enUS");
        self.u32((text.len() * 2) as u32);
        self.u32(28);
        for &byte in text {
            self.u16(u16::from(byte));
        }
    }
    pub fn xyz(&mut self, values: [f32; 3]) -> Result<()> {
        self.append(b"XYZ ");
        self.u32(0);
        for value in values {
            self.fixed(value)?;
        }
        Ok(())
    }
    pub fn matrix(&mut self, values: Matrix) -> Result<()> {
        self.append(b"sf32");
        self.u32(0);
        for value in values.into_iter().flatten() {
            self.fixed(value)?;
        }
        Ok(())
    }
    pub fn cicp(&mut self, values: [u8; 4]) {
        self.append(b"cicp");
        self.u32(0);
        self.append(&values);
    }
    pub fn para(&mut self, kind: u16, values: &[f32]) -> Result<()> {
        self.append(b"para");
        self.u32(0);
        self.u16(kind);
        self.u16(0);
        for &value in values {
            self.fixed(value)?;
        }
        Ok(())
    }
    pub fn finish(mut self) -> Vec<u8> {
        use md5::{Digest, Md5};
        // ICC IDs exclude flags, rendering intent and the ID field itself.
        let mut hash = Md5::new();
        hash.update(&self.bytes[..44]);
        hash.update([0; 4]);
        hash.update(&self.bytes[48..64]);
        hash.update([0; 4]);
        hash.update(&self.bytes[68..84]);
        hash.update([0; 16]);
        hash.update(&self.bytes[100..]);
        self.bytes[84..100].copy_from_slice(&hash.finalize());
        self.bytes
    }
}
