# Upstream boundary

## Decision

The production repository is standalone and consumes JPEG XL crates through Cargo. It does not
carry `jxl-rs` as workspace members, a subtree, or a submodule.

The currently published `jxl` API returns completed CPU output. That is too late for this
GPU-required codec and is not used as a production fallback. If upstream integration is pursued in
the future, a correct pre-IDCT backend would need four narrowly scoped capabilities:

1. optional backend injection before a frame render pipeline is constructed;
2. an owned, versioned packet for decoded groups and borrowed pre-IDCT coefficient export;
3. coordinator-only submission after the parallel decode runner reaches its barrier; and
4. terminal output negotiation so native YUV/NV12 can replace the final RGB Save without an RGB
   readback.

None of these capabilities should expose `wgpu` types. The protocol and device implementation stay
in this repository, while an upstream adapter only translates decoder semantics into
`jxl_gpu_protocol` values.

## Why not drive `jxl` implementation modules directly?

`jxl` 0.6 exposes useful header, bit-reader, entropy, and frame modules, but the complete state
needed for an external driver is not a supported public boundary. In particular, `DecoderState`
keeps the file header/reference/LF frames private, `Frame` keeps LF/HF global state and metadata
private, and the render pipeline builder/trait are crate-private. The public group decoder already
performs the CPU transform into pixels. Copying the parser and group orchestration would be a second
decoder, not an adapter.

The helper crates do not close that gap: `jxl_transforms` and `jxl_simd` implement CPU math,
`jxl_cms` handles color management, and `jxl_macros` supports the implementation. They remain
normal transitive dependencies of `jxl`; the GPU backend should depend on them directly only for a
specific scalar oracle or compatibility test.

The audit was performed against the crates.io `0.6.0` family, all built from upstream commit
`fbed310bda2496c97672f7f427ca7a2aebe035d4`:

- public bit-reader, ANS, Huffman, hybrid-uint, header, TOC, and transform primitives can be reused;
- `FullModularImage` can be driven only by recreating substantial frame orchestration outside the
  normal decoder;
- the semantic VarDCT group decoder, block-context map, coefficient order, dequant matrices, and
  LF/HF state needed to produce a GPU packet are not a supported external boundary;
- the public output contract contains packed gray/RGB/BGR variants, not caller-owned GPU buffers
  or planar video output.

Building a second decoder from the public primitives would therefore duplicate container parsing,
progressive passes, reference frames, animation blending, and scheduling. That is a materially
larger and less stable design than one narrow upstream packet sink.

## Repository layers

- **GPU codec frontend:** bounded independent container/header/group orchestration; unsupported
  profiles reject without a CPU codec path.
- **Protocol/backend:** independent of `jxl`; shared by the GPU encoder, decoder, captures, and
  tests.
- **Reference oracle:** published `jxl` and `cjxl`/`djxl` may be dev dependencies or external
  harness processes only.
- **Upstream adapter:** intentionally small and versioned separately when the required hooks exist.
  Unsupported stages must reject before submission; no CPU RGB roundtrip may be described as native
  decode.

The source-tree prototype remains available only as Git history for developing and upstreaming that
adapter.

## Minimal upstream contract

The preferred future hook is an owned/event-oriented packet sink, not public access to every
decoder field:

```rust,ignore
trait DecodePacketSink {
    fn begin_frame(&mut self, metadata: &FramePacketMetadata) -> Result<Decision, Error>;
    fn modular_plane(&mut self, plane: ModularPlaneRef<'_>) -> Result<(), Error>;
    fn vardct_group(&mut self, group: VarDctGroupRef<'_>) -> Result<(), Error>;
    fn end_frame(&mut self) -> Result<(), Error>;
}
```

The VarDCT packet must contain transform coverage, quantized coefficients, progressive-pass
completeness, LF/DC planes, quant/dequant information, color-correlation data, and EPF metadata.
The decoder remains responsible for container, reference-frame, and dependency scheduling; the
backend receives a versioned semantic packet at the point immediately before CPU dequant/IDCT.

Terminal output negotiation is separate. It lets the backend own the final pitch-linear buffer or
display texture instead of forcing a CPU RGB buffer, while keeping all `wgpu` types outside
upstream. This standalone repository does not wait for those hooks: its own GPU frontend exposes
only profiles it can execute end to end and rejects the rest.
