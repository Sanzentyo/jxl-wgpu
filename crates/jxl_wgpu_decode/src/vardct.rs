//! Standard VarDCT frontend and GPU-resident artifact interfaces.
//!
//! [`frontend`] inventories a deliberately bounded standard JPEG XL VarDCT
//! profile and parses metadata prefixes without consuming image entropy on the
//! host. [`artifact`] defines the exact, bytemuck-safe storage ABI used to turn
//! GPU-reconstructed HF metadata into transform tasks, indirect dispatches, and
//! a GPU coefficient sink.
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
