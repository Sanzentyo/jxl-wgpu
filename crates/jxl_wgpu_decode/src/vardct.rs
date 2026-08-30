//! Standard VarDCT frontend and GPU-resident artifact interfaces.
//!
//! [`frontend`] inventories a deliberately bounded standard JPEG XL VarDCT
//! profile and parses metadata prefixes without consuming image entropy on the
//! host. [`packet`] decodes the accepted image entropy on the GPU and [`artifact`]
//! defines the exact, bytemuck-safe storage ABI used to turn
//! GPU-reconstructed HF metadata into transform tasks, indirect dispatches, and
//! a GPU coefficient sink. [`output`] packs resident XYB planes as sRGB8 without
//! a CPU pixel path.
//!
//! These modules are low-level building blocks. Applications normally reach
//! them through the crate's decoder session API; they are public so alternate
//! runtime-neutral submission engines can share the same negotiated profile and
//! resident-buffer ABI without copying private structs.

/// GPU-resident task, dispatch, and coefficient-sink ABI.
pub mod artifact {
    pub use crate::vardct_artifact::*;
}

/// Bounded standard-codestream inventory and metadata-prefix IR.
pub mod frontend {
    pub use crate::vardct_frontend::*;
}

/// Strict zero-AC regular-VarDCT packet profile and GPU entropy ABI.
pub mod packet {
    pub use crate::vardct_packet::*;
}

/// Resident XYB-to-sRGB8 GPU output packing.
pub mod output {
    pub use crate::vardct_output::*;
}

/// Adaptive LF smoothing ABI shared by standard VarDCT submission engines.
pub mod lf {
    pub use crate::vardct_lf::*;
}

/// Strict zero-AC resource table and LF dequantization ABI.
pub mod resource {
    pub use crate::vardct_resource::*;
}
