# GPU architecture

## Repository boundary

This repository is a standalone Cargo workspace. Production codec crates do not vendor, patch, or
call an upstream CPU codec. Published `jxl` and reference `cjxl`/`djxl` binaries are dev-only
correctness oracles.
The backend-neutral protocol, portable image formats, `wgpu` execution, stable decode facade, and
measurement harness can therefore be released independently.

The dependency graph is intentionally one-way:

```text
jxl_gpu_bitstream   jxl_gpu_protocol   jxl_gpu_formats
        |                   |                  |
        +-------------------+------------------+
                            v
                         jxl_wgpu
                         /      \
                        v        v
              jxl_wgpu_decode  jxl_wgpu_encode
                        \        /
                         harness
```

`jxl_gpu_protocol`, `jxl_gpu_bitstream`, and `jxl_gpu_formats` contain no `wgpu` or upstream-decoder
types. `jxl_wgpu` has no production dependency on `jxl`; reference codec crates and tools are used
only by tests and the harness.

## GPU-required codec path

Container/header parsing and packet ordering remain host orchestration. There is no host pixel
codec and no production fallback to `jxl`, `cjxl`, or `djxl`:

```text
JPEG XL bytes -> bounded container/header parse -> GPU group decode
                                                -> GPU restoration/color/output
                                                |-> optional CPU readback
                                                `-> same-queue display/consumer

GPU image source -> GPU prediction/transform/quantization/tokenization
                 -> deterministic group packet assembly -> JPEG XL codestream/container
```

An unsupported profile rejects during capability negotiation or header validation. A CPU oracle
may decode the result in tests, but oracle code cannot be reached from a production encode/decode
session. The initial interoperable target is a deliberately narrow single-group lossless Modular
profile; capabilities expand only with valid-bitstream round trips and conformance evidence.

## Protocol execution

For capture/replay tests and a future decoder bridge, `jxl_gpu_protocol` describes planes,
resources, decoded group payloads, VarDCT packets, operations, output descriptors, changed regions,
and transactional frame sessions. `jxl_wgpu` validates and lowers this protocol into a bounded
batch:

```text
RenderPlan
    -> validation and capability negotiation
    -> lifetime-based resident-buffer allocation
    -> kernel selection and safe fusion
    -> one command encoder / ordered queue submission
    -> GPU-resident output or explicit mapped readback
```

Unsupported nodes reject before output becomes authoritative. Production codec sessions do not
fall back to a CPU codec, and the backend never returns a partially valid frame as success.

## Portable output formats

`jxl_gpu_formats` separates the independent properties of an image:

- color model and channel semantics;
- component/sample representation and bit depth;
- interleaved, planar, semi-planar, or packed plane organization;
- chroma subsampling and siting;
- matrix coefficients, primaries, transfer, and full/limited range;
- checked per-plane offset, extent, row stride, and logical byte size.

This models all relevant pitch-linear NVIDIA VPI 4.1 predefined formats as well as common planar
and higher-bit-depth video formats. CUDA-specific block-linear storage is deliberately out of
scope: WebGPU does not expose that physical layout, and this project does not pretend that a
portable buffer is CUDA block-linear memory.

Every layout calculation is checked for overflow, overlapping planes, undersized strides, and
subsampling constraints. Odd extents use explicit ceil division where the format permits them;
formats such as packed 4:2:2 can require an even width.

## GPU output and display

There are three distinct terminal paths:

1. CPU output: map a readback buffer after a submission token completes.
2. GPU buffer output: return reference-counted pitch-linear plane buffers immediately after queue
   submission.
3. Display texture output: encode conversion/copy commands after decode work on the same queue and
   return an RGBA texture suitable for sampling, rendering, or copying to a surface texture.

Queue ordering is the synchronization primitive for path 2 and 3. No host wait is required before
encoding a dependent command. Resource handles keep allocations alive; explicit completion is
needed only for CPU access or slot reuse across unrelated queues/devices.

`DisplayPipeline` caches pipelines and bind-group layouts by source format and color conversion.
Direct RGBA copies are used when storage and texture layouts agree. Planar, semi-planar, and packed
YUV use a shader conversion into an RGBA display texture. WebGPU has no portable native NV12
multi-plane texture, so the public contract remains explicit plane buffers rather than claiming a
native multi-plane texture object.

## Animation and concurrency

Sync and async animation frontends drive one GPU codec state machine. A frame contains its index,
duration/tick metadata, presentation timestamp, loop metadata, and composed output. Reference-frame
and blend dependencies are explicit session state; a frame slot is not reusable while a later GPU
submission or caller lease still depends on it.

The synchronous API advances and, when requested, waits for one GPU frame at a time. The
runtime-neutral async API is expressed with `Future`, `Poll`, `Context`, and `Waker`; it does not
depend on Tokio, async-std, or a particular reactor. Completion callbacks wake the task.

GPU animation output uses a bounded in-flight budget. A slot cannot be reused until its consumer
lease is released or its submission is complete. This gives explicit backpressure and prevents an
animation, a sequential batch, or concurrent decoders from growing GPU memory without limit.

Repeated decodes share immutable shader modules, pipeline caches, and a bounded buffer pool.
Concurrent decoders may share a device and queue while retaining separate frame state. The harness
reports both per-image latency and aggregate throughput for small, large, sequential, concurrent,
and animation workloads.

## Safety invariants

- All sizes, offsets, strides, dispatch counts, and allocation sums use checked arithmetic.
- A plane descriptor cannot address outside its allocation or overlap another writable plane.
- A protocol resource revision increases monotonically and a final submission requires complete
  latest group revisions.
- GPU-resident public output is never returned from an internal recyclable buffer.
- CPU mapping waits for exactly the associated submission; GPU consumers use ordered queue work.
- Device/resource-limit errors remain typed and never downgrade precision silently.
- Production crates deny unsafe Rust.
