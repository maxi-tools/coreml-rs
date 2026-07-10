#![allow(clippy::all)]
use coreml_rs_fork::{compute_plan_device_counts, ComputePlanDeviceCounts, ComputePlatform};

#[test]
fn compute_plan_bogus_path_returns_none() {
    let counts = compute_plan_device_counts(
        "/nonexistent/definitely-not-a-model.mlmodelc",
        ComputePlatform::All,
    );
    assert!(counts.is_none());
}

#[test]
fn ane_fraction_math() {
    let counts = ComputePlanDeviceCounts {
        total: 10,
        ane: 8,
        gpu: 0,
        cpu: 2,
    };
    assert!((counts.ane_fraction() - 0.8).abs() < 1e-9);

    let empty = ComputePlanDeviceCounts {
        total: 0,
        ane: 0,
        gpu: 0,
        cpu: 0,
    };
    assert_eq!(empty.ane_fraction(), 0.0);
}
