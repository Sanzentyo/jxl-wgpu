// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fmt;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use jxl_gpu_protocol::{BackendCapabilities, BackendError, FrameSession, RenderBackend};
use jxl_gpu_protocol::{FrameSessionDesc, RenderPlan};

use crate::buffer_pool::{BufferPool, WgpuBufferPoolStats};
#[cfg(not(target_arch = "wasm32"))]
use crate::capability::capabilities;
use crate::pipeline_cache::PipelineCache;
use crate::session::WgpuFrameSession;
use crate::{Error, Planner, Result};

#[derive(Clone, Debug)]
pub struct WgpuMemoryPolicy {
    /// Maximum aggregate bytes in the physical resident plane slots. Readback, per-submission
    /// VarDCT packet buffers, and immutable kernel parameter buffers are transient command
    /// resources and are not included.
    pub max_resident_bytes: u64,
    /// Maximum simultaneously live bytes assigned to intermediate planes inside the arena.
    pub max_scratch_bytes: u64,
    /// Maximum bytes allocated by one submission for explicit transient GPU buffers, including
    /// VarDCT packet uploads, immutable kernel tables, packed outputs, and readback staging.
    pub max_transient_bytes: u64,
    /// Maximum idle bytes retained by the backend-wide internal buffer pool. This is separate
    /// from the live resident and transient submission budgets; set it to zero to disable reuse.
    pub max_cached_buffer_bytes: u64,
    /// Reserved for the future tiled scheduler. `Auto` still selects resident execution today;
    /// explicit `MemoryMode::Streaming` requests return a typed unsupported error.
    pub prefer_streaming: bool,
}

impl Default for WgpuMemoryPolicy {
    fn default() -> Self {
        Self {
            max_resident_bytes: 512 * 1024 * 1024,
            max_scratch_bytes: 256 * 1024 * 1024,
            max_transient_bytes: 256 * 1024 * 1024,
            max_cached_buffer_bytes: 128 * 1024 * 1024,
            prefer_streaming: false,
        }
    }
}

/// Native policy for mapping the final storage allocation directly on the CPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DirectReadbackPolicy {
    /// Enable direct mapping only on adapters reported as integrated or CPU devices.
    #[default]
    Auto,
    /// Always use the portable storage-to-staging copy.
    Disabled,
    /// Request direct mapping on any adapter and fail adapter creation if the feature is missing.
    Force,
}

#[derive(Clone, Debug)]
pub struct WgpuBackendConfig {
    pub label: String,
    pub power_preference: wgpu::PowerPreference,
    pub memory: WgpuMemoryPolicy,
    /// Chooses safe automatic UMA mapping, portable staging, or explicitly forced direct mapping.
    pub direct_readback_policy: DirectReadbackPolicy,
    pub enable_timestamps: bool,
    pub strict_features: bool,
}

impl Default for WgpuBackendConfig {
    fn default() -> Self {
        Self {
            label: "jxl-wgpu".into(),
            power_preference: wgpu::PowerPreference::HighPerformance,
            memory: WgpuMemoryPolicy::default(),
            direct_readback_policy: DirectReadbackPolicy::Auto,
            enable_timestamps: true,
            strict_features: false,
        }
    }
}

#[derive(Clone)]
pub struct WgpuBackend {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) info: wgpu::AdapterInfo,
    pub(crate) config: WgpuBackendConfig,
    pub(crate) pipelines: Arc<PipelineCache>,
    pub(crate) buffers: Arc<BufferPool>,
}

impl fmt::Debug for WgpuBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WgpuBackend")
            .field("adapter", &self.info)
            .field("config", &self.config)
            .field("pipeline_cache_empty", &self.pipelines.is_empty())
            .field("buffer_pool", &self.buffers.stats())
            .finish_non_exhaustive()
    }
}

impl WgpuBackend {
    pub fn from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        info: wgpu::AdapterInfo,
        config: WgpuBackendConfig,
    ) -> Result<Self> {
        if config.enable_timestamps
            && config.strict_features
            && !device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return Err(Error::Unsupported(
                "TIMESTAMP_QUERY was requested but is unavailable on the supplied device".into(),
            ));
        }
        #[cfg(not(target_arch = "wasm32"))]
        if direct_readback_requested(&config, info.device_type)
            && config.direct_readback_policy == DirectReadbackPolicy::Force
            && !device
                .features()
                .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS)
        {
            return Err(Error::Unsupported(
                "direct readback was forced but the supplied device lacks MAPPABLE_PRIMARY_BUFFERS"
                    .into(),
            ));
        }
        #[cfg(target_arch = "wasm32")]
        if config.direct_readback_policy == DirectReadbackPolicy::Force {
            return Err(Error::Unsupported(
                "direct readback cannot be forced on browser WebGPU".into(),
            ));
        }
        let buffers = BufferPool::new(device.clone(), config.memory.max_cached_buffer_bytes);
        Ok(Self {
            device,
            queue,
            info,
            config,
            pipelines: Arc::new(PipelineCache::default()),
            buffers,
        })
    }

    pub async fn request_default(config: WgpuBackendConfig) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: config.power_preference,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|_| Error::NoAdapter)?;
        let info = adapter.get_info();
        let adapter_features = adapter.features();
        let mut required_features = wgpu::Features::empty();
        if config.enable_timestamps && adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        } else if config.enable_timestamps && config.strict_features {
            return Err(Error::Unsupported(
                "TIMESTAMP_QUERY was requested but is unavailable".into(),
            ));
        }
        // Unified-memory native adapters can map the final storage buffer directly. This avoids a
        // full output copy while preserving the portable staging path on other adapters.
        #[cfg(not(target_arch = "wasm32"))]
        if direct_readback_requested(&config, info.device_type) {
            if adapter_features.contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS) {
                required_features |= wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
            } else if config.direct_readback_policy == DirectReadbackPolicy::Force {
                return Err(Error::Unsupported(
                    "direct readback was forced but MAPPABLE_PRIMARY_BUFFERS is unavailable".into(),
                ));
            }
        }
        let required_limits = wgpu::Limits::default().using_resolution(adapter.limits());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some(&config.label),
                required_features,
                required_limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;
        Self::from_device(device, queue, info, config)
    }

    pub const fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub const fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub const fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    /// Whether CPU submissions can use direct mapping on this configured device.
    pub fn direct_readback_enabled(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            direct_readback_requested(&self.config, self.info.device_type)
                && self
                    .device
                    .features()
                    .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS)
        }
        #[cfg(target_arch = "wasm32")]
        {
            false
        }
    }

    /// Drops lazily compiled pipelines, for example after a device-specific tuning reset.
    pub fn clear_pipeline_cache(&self) {
        self.pipelines.clear();
    }

    /// Drops idle internal buffers. In-flight submissions and caller-owned GPU outputs are not
    /// part of this cache and remain valid.
    pub fn clear_buffer_pool(&self) {
        self.buffers.clear();
    }

    /// Returns allocation reuse and current idle-cache occupancy counters.
    pub fn buffer_pool_stats(&self) -> WgpuBufferPoolStats {
        self.buffers.stats()
    }

    pub fn create_session(
        &self,
        frame: &FrameSessionDesc,
        plan: Arc<RenderPlan>,
    ) -> Result<WgpuFrameSession> {
        plan.validate()?;
        let execution =
            Planner::new(self.device.limits(), self.config.memory.clone()).plan(frame, &plan)?;
        WgpuFrameSession::new(self.clone(), frame.clone(), plan, execution)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn direct_readback_requested(config: &WgpuBackendConfig, device_type: wgpu::DeviceType) -> bool {
    match config.direct_readback_policy {
        DirectReadbackPolicy::Disabled => false,
        DirectReadbackPolicy::Force => true,
        DirectReadbackPolicy::Auto => matches!(
            device_type,
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu
        ),
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl RenderBackend for WgpuBackend {
    fn capabilities(&self) -> BackendCapabilities {
        capabilities(&self.device, &self.info)
    }

    fn create_frame_session(
        &self,
        frame: &FrameSessionDesc,
        plan: Arc<RenderPlan>,
    ) -> std::result::Result<Box<dyn FrameSession>, BackendError> {
        self.create_session(frame, plan)
            .map(|session| Box::new(session) as Box<dyn FrameSession>)
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(target_arch = "wasm32"))]
    use super::direct_readback_requested;
    use super::{DirectReadbackPolicy, WgpuBackendConfig};

    #[test]
    fn automatic_direct_readback_is_the_default_policy() {
        assert_eq!(
            WgpuBackendConfig::default().direct_readback_policy,
            DirectReadbackPolicy::Auto
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn automatic_direct_readback_is_limited_to_unified_memory_devices() {
        let config = WgpuBackendConfig::default();
        assert!(direct_readback_requested(
            &config,
            wgpu::DeviceType::IntegratedGpu
        ));
        assert!(direct_readback_requested(&config, wgpu::DeviceType::Cpu));
        assert!(!direct_readback_requested(
            &config,
            wgpu::DeviceType::DiscreteGpu
        ));

        let forced = WgpuBackendConfig {
            direct_readback_policy: DirectReadbackPolicy::Force,
            ..config
        };
        assert!(direct_readback_requested(
            &forced,
            wgpu::DeviceType::DiscreteGpu
        ));

        let disabled = WgpuBackendConfig {
            direct_readback_policy: DirectReadbackPolicy::Disabled,
            ..forced
        };
        assert!(!direct_readback_requested(
            &disabled,
            wgpu::DeviceType::IntegratedGpu
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn supplied_device_must_satisfy_forced_and_strict_features() {
        fn test_device() -> Option<(wgpu::Device, wgpu::Queue, wgpu::AdapterInfo)> {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::None,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                    apply_limit_buckets: false,
                }))
                .ok()?;
            let info = adapter.get_info();
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("jxl-wgpu supplied feature invariant test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))
            .ok()
            .map(|(device, queue)| (device, queue, info))
        }

        let Some((device, queue, info)) = test_device() else {
            eprintln!("skipping supplied-device feature test: no adapter");
            return;
        };
        let forced = WgpuBackendConfig {
            enable_timestamps: false,
            direct_readback_policy: DirectReadbackPolicy::Force,
            ..WgpuBackendConfig::default()
        };
        assert!(matches!(
            super::WgpuBackend::from_device(device, queue, info, forced),
            Err(crate::Error::Unsupported(message))
                if message.contains("MAPPABLE_PRIMARY_BUFFERS")
        ));

        let Some((device, queue, info)) = test_device() else {
            eprintln!("skipping supplied-device feature test: no second adapter request");
            return;
        };
        let strict_timestamps = WgpuBackendConfig {
            enable_timestamps: true,
            strict_features: true,
            direct_readback_policy: DirectReadbackPolicy::Disabled,
            ..WgpuBackendConfig::default()
        };
        assert!(matches!(
            super::WgpuBackend::from_device(device, queue, info, strict_timestamps),
            Err(crate::Error::Unsupported(message)) if message.contains("TIMESTAMP_QUERY")
        ));
    }
}
