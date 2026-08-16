//! Dependency-free image writers (PPM P6 + PNG).
//!
//! Both emit rows BOTTOM-TO-TOP: the render buffers use `origin="lower"`
//! (row 0 = bottom of frame, matching the reference), while PPM/PNG define
//! row 0 as the top scanline.
//!
//! The PNG path uses *stored* (uncompressed) deflate blocks — legal per
//! RFC 1951 — so no compression library is needed. Files are larger than
//! zlib-compressed PNGs but open everywhere, including VS Code's previewer.

use anyhow::Result;
use shannon_core::Vec3;

fn quantize(pixels: &[Vec3], width: u32, height: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in (0..height).rev() {
        for x in 0..width {
            let p = pixels[(y * width + x) as usize];
            rgb.push((p.x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            rgb.push((p.y.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            rgb.push((p.z.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }
    rgb
}

/// Binary PPM (P6).
pub fn write_ppm(path: &str, pixels: &[Vec3], width: u32, height: u32) -> Result<()> {
    let rgb = quantize(pixels, width, height);
    let mut out = Vec::with_capacity(20 + rgb.len());
    out.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
    out.extend_from_slice(&rgb);
    std::fs::write(path, out)?;
    Ok(())
}

/// 8-bit RGB PNG with stored deflate blocks. No dependencies.
pub fn write_png(path: &str, pixels: &[Vec3], width: u32, height: u32) -> Result<()> {
    let rgb = quantize(pixels, width, height);

    // Filtered scanlines: filter byte 0 (None) + row data.
    let stride = (width * 3) as usize;
    let mut scan = Vec::with_capacity(rgb.len() + height as usize);
    for row in rgb.chunks(stride) {
        scan.push(0u8);
        scan.extend_from_slice(row);
    }

    // zlib stream: header + stored deflate blocks + adler32.
    let mut z = vec![0x78u8, 0x01];
    for (i, block) in scan.chunks(65_535).enumerate() {
        let last = (i + 1) * 65_535 >= scan.len();
        z.push(if last { 1 } else { 0 });
        let len = block.len() as u16;
        z.extend_from_slice(&len.to_le_bytes());
        z.extend_from_slice(&(!len).to_le_bytes());
        z.extend_from_slice(block);
    }
    z.extend_from_slice(&adler32(&scan).to_be_bytes());

    let mut png = Vec::with_capacity(z.len() + 64);
    png.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit, RGB, deflate, none, non-interlaced
    push_chunk(&mut png, b"IHDR", &ihdr);
    push_chunk(&mut png, b"IDAT", &z);
    push_chunk(&mut png, b"IEND", &[]);
    std::fs::write(path, png)?;
    Ok(())
}

fn push_chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    let mut crc = Crc32::new();
    crc.update(tag);
    crc.update(data);
    out.extend_from_slice(&crc.finish().to_be_bytes());
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65_521;
        b = (b + a) % 65_521;
    }
    (b << 16) | a
}

struct Crc32 {
    table: [u32; 256],
    value: u32,
}

impl Crc32 {
    fn new() -> Self {
        let mut table = [0u32; 256];
        for (n, slot) in table.iter_mut().enumerate() {
            let mut c = n as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *slot = c;
        }
        Self {
            table,
            value: 0xFFFF_FFFF,
        }
    }
    fn update(&mut self, data: &[u8]) {
        for &b in data {
            self.value = self.table[((self.value ^ b as u32) & 0xFF) as usize] ^ (self.value >> 8);
        }
    }
    fn finish(&self) -> u32 {
        self.value ^ 0xFFFF_FFFF
    }
}
