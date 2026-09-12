use super::super::tests::{backend, drain, line_program};
use super::*;

#[test]
fn geometry_limits_and_replay_allocation_fail_without_publishing_a_cache() {
    let backend = backend();
    for resource in [
        SplineResource::GeometrySteps,
        SplineResource::DrawRecords,
        SplineResource::TileReferences,
    ] {
        let mut plan = GeometryPlan::new(
            line_program(&backend, 2),
            Extent2d::new(64, 32),
            [0.0, 1.0],
            &backend.device().limits(),
        )
        .unwrap();
        match resource {
            SplineResource::GeometrySteps => plan.params.work[0] = 1,
            SplineResource::DrawRecords => plan.params.run[2] = 1,
            SplineResource::TileReferences => plan.params.run[3] = 1,
            _ => unreachable!(),
        }
        let error = plan.submit(backend.clone()).unwrap().wait().unwrap_err();
        assert!(
            matches!(error, Error::SplineResourceLimit { resource: actual, limit: 1 } if actual == resource),
            "{error:?}"
        );
        drain(&backend, 0);
    }
    let plan = GeometryPlan::new(
        line_program(&backend, 2),
        Extent2d::new(64, 32),
        [0.0, 1.0],
        &backend.device().limits(),
    )
    .unwrap();
    let mut pending = plan.submit(backend.clone()).unwrap();
    let budget = backend.transient_memory_budget();
    let blocker = budget
        .try_reserve(budget.snapshot().available_bytes)
        .unwrap();
    let error = loop {
        match pending.complete_submission() {
            Ok(None) => {}
            Ok(Some(_)) => panic!("cache allocated without admission"),
            Err(error) => break error,
        }
    };
    assert!(matches!(error, Error::MemoryBackpressure(_)), "{error:?}");
    assert!(pending.expected.is_none());
    drop((pending, blocker));
    drain(&backend, 0);
}
