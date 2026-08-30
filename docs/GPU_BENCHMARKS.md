# GPU performance evidence

This document defines the evidence required for performance claims. A result is accepted only
after output comparison and path verification pass. Production reports cover GPU-required
encode/decode; development oracles are labelled separately and cannot select a production codec
path.

## Workloads

The harness measures the cases that matter for application policy instead of reducing a workload
to image dimensions alone:

- one small still image;
- one medium or large still image;
- a warm sequential batch sharing one device, queue, pipelines, uploads, and readback pool;
- bounded concurrent decoders, reporting per-image latency and aggregate throughput;
- animation playback with bounded in-flight frames;
- GPU-resident display output without readback; and
- direct-map and staged-copy CPU-readback controls.

Each workload records warmup, sample count, dimensions, encoded/output bytes, format, codec path,
adapter fingerprint, submission count, readback mode, latency percentiles, throughput, and peak
explicit GPU bytes. Device creation and shader compilation are reported separately from
steady-state measurements.

Policy data uses the complete workload key:

```text
(adapter, codec profile, output format, size bucket, batch depth, concurrency, readback/display)
    -> validated GPU kernel/batch/readback policy or typed unsupported
```

An unknown key uses a validated general GPU variant or returns a typed rejection.

## CPU-readback evidence

CPU readback is a transport after GPU codec and conversion work. A CPU-readback result is reported
as favorable only when it:

1. passes the format-specific accuracy contract;
2. executes and verifies the expected GPU submissions;
3. improves the target workload's median without exceeding its p95 guardrail;
4. includes upload, conversion, mapping, row de-padding, and output allocation in the measured
   interval; and
5. reproduces across enough interleaved runs to survive ordering and thermal bias.

Small-image latency, warm batches, concurrent independent sessions, and animation are separate
results. Batching can cover several images with one command buffer and one completion wait, while
animation batches remain bounded by presentation latency and the configured in-flight count.
GPU-resident display is measured independently because it removes mapping and transfer costs.

Direct-map and staged-copy measurements must identify the adapter memory architecture and enabled
features. The report includes `WgpuSubmissionStats`, `ImageReadbackStats`, pending transient bytes,
and buffer-pool counters so allocation reuse cannot be mistaken for kernel improvement.

## Report lifecycle

Checked-in tuning data contains an adapter fingerprint and the codec profile, shader, format
layout, driver, and `wgpu` revisions that produced it. A change to any of those inputs invalidates
the affected entry until it is measured again. Performance claims require a current standalone
report for the exact workload key.
