use super::{backend, corpus, frames, oracle};
use jxl_gpu_formats::{
    ColorSpace, ColorSpecification, PixelFormat, RgbChannelOrder, TransferFunction,
};
use jxl_gpu_protocol::icc::{IccProfile, IccRenderingIntent};
use jxl_test_support::oracles::color;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuOutputRequest};
use std::path::Path;

mod embedded;
mod pcs;
mod profiles;
