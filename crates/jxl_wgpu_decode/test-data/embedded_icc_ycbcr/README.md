# YCbCr and original ICC presentation references

These 60 files are generated independently from existing Modular/VarDCT device
references and the exact embedded_icc RGB/Gray profiles. Each line is a binary32
word in hexadecimal; pixels are interleaved with alpha. For each of five sources,
linear, srgb, and other provide scalar centers, native values, and lower/upper
GPU intervals.

See ../embedded_icc_ycbcr_generator/README.md for reproducible generation, source
provenance, profile-domain behavior, and the derivation of each error interval.
