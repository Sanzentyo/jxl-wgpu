use crate::report::AdapterReport;

pub fn enumerate_adapters() -> Vec<AdapterReport> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    let mut reports = adapters
        .into_iter()
        .map(|adapter| {
            let info = adapter.get_info();
            let limits = adapter.limits();
            AdapterReport {
                name: info.name,
                vendor: info.vendor,
                device: info.device,
                device_type: format!("{:?}", info.device_type),
                backend: format!("{:?}", info.backend),
                driver: info.driver,
                driver_info: info.driver_info,
                pci_bus_id: info.device_pci_bus_id,
                subgroup_min_size: info.subgroup_min_size,
                subgroup_max_size: info.subgroup_max_size,
                features: format!("{:?}", adapter.features()),
                max_buffer_size: limits.max_buffer_size,
                max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                max_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
                max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            }
        })
        .collect::<Vec<_>>();
    reports.sort_by(|left, right| {
        (&left.backend, &left.name, left.vendor, left.device).cmp(&(
            &right.backend,
            &right.name,
            right.vendor,
            right.device,
        ))
    });
    reports
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_enumeration_is_safe_without_an_adapter() {
        let adapters = enumerate_adapters();
        assert!(adapters.iter().all(|adapter| !adapter.name.is_empty()));
    }
}
