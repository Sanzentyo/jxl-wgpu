//! Host-shareable control ABI for the common GPU entropy executor.

/// Bit bounds and LZ77 ring shape consumed directly by `modular_entropy.wgsl`.
///
/// The consumer supplies storage access and its LZ77 scratch base as WGSL functions. Image
/// geometry, prediction, output, and coefficient state remain outside this prefix.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct EntropyStreamParams {
    pub token_start: u32,
    pub token_end: u32,
    pub lz77_window_mask: u32,
}

const _: () = {
    assert!(std::mem::size_of::<EntropyStreamParams>() == 12);
    assert!(std::mem::align_of::<EntropyStreamParams>() == 4);
};
