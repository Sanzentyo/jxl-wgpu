//! Capture, replay, verification, benchmarking, and tuning support for `jxl_wgpu`.

#![deny(unsafe_code)]

pub mod adapter;
pub mod benchmark;
pub mod capture;
pub mod codec;
pub mod compare;
pub mod config;
pub mod error;
pub mod reference;
pub mod replay;
pub mod report;
pub mod synthetic;
pub mod tune;

/// Capture file schema version produced by this crate.
pub const CAPTURE_SCHEMA_VERSION: u16 = 1;

pub use error::{Error, Result};
