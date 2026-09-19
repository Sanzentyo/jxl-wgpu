# Documentation index

Choose the document for the question at hand. The large roadmap, corpus, and memory
references are intended to be consulted by relevant heading or feature, not read
in full before each edit.

## Entry points

| Need | Start here |
|---|---|
| Run the project or understand its scope | [Project README](../README.md) |
| Change code and select validation | [Contributing](../CONTRIBUTING.md) |
| Repository-specific agent boundaries | [AGENTS.md](../AGENTS.md) |
| Decide whether a feature is supported or complete | [Full JPEG XL roadmap](FULL_JPEG_XL_ROADMAP.md) |
| Use the public decoder or encoder | [Decoder](../crates/jxl_wgpu_decode/README.md) / [encoder](../crates/jxl_wgpu_encode/README.md) |
| Use capture/replay, codec workloads, or test helpers | [Harness](../tools/jxl_gpu_harness/README.md) / [test support](../tools/jxl_test_support/README.md) |

## Topic references

| Topic | Reference |
|---|---|
| Backend boundaries, scheduling, and GPU output | [GPU architecture](GPU_ARCHITECTURE.md) |
| Encoder stages and production responsibilities | [Encoder architecture](ENCODER_ARCHITECTURE.md) |
| Shader ABI, binding, layout, and resource lifetime | [WGSL memory](WGSL_MEMORY.md) |
| CPU-oracle/dependency separation and provenance | [Upstream boundary](UPSTREAM_BOUNDARY.md) |
| Fixture families, reproduction, and conformance evidence | [Conformance corpus](CONFORMANCE_CORPUS.md) |
| Measured performance and workload contracts | [GPU benchmarks](GPU_BENCHMARKS.md) |
| Portable output formats and precision policies | [VPI format coverage](VPI_FORMAT_COVERAGE.md) / [format APIs](../crates/jxl_gpu_formats/README.md) |
| Opaque metadata ownership, limits, and writing | [Container metadata](CONTAINER_METADATA.md) |
| ICC profiles, device output, and supported connections | [ICC color](ICC_COLOR.md) |
| Luminance adaptation | [Tone mapping](TONE_MAPPING.md) |
| Requested RGB gamut mapping | [Gamut mapping](GAMUT_MAPPING.md) |
| Still gain maps, headroom, and reference white | [Gain maps](GAIN_MAP.md) |

## Which source to update

The roadmap owns capability status and acceptance gates. Crate READMEs describe
public behavior; topic documents explain the corresponding contracts. Keep the
root README as an overview rather than duplicating every corpus result there.
The contributor guide explains validation and evidence updates.

The [pre-reorganization README](../README.legacy.md) is a historical snapshot of
implementation notes at `f59ca9ce20123c7d3d9cf270e959aff456db6802`, not current
capability policy or agent instructions. It remains at the repository root so its
relative links keep their original meaning. New status and evidence belong in the
canonical documents above, not in that snapshot.
