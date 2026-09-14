# Third-party notices

`src/vardct/quantization.rs` expands default quantization constants and formulas from the
BSD-3-Clause libjxl 0.12.0 `lib/jxl/quant_weights.cc` at commit
`a7a9c787341cf703dede03c2009fa460cae5e5df`. The natural-order contract follows
`lib/jxl/ac_strategy.cc`. Only bounded metadata is evaluated here; production image processing
remains on GPU. Native metadata in `test-data/vardct_metadata.bin` is an independent test oracle.

Copyright (c) the JPEG XL Project Authors. All rights reserved. The BSD-3-Clause terms are
reproduced in this crate's `LICENSE` file.
