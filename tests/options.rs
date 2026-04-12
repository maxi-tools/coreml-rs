use coreml_rs_fork::CoreMLModelOptions;

#[test]
fn test_coreml_model_options_default() {
    let opts = CoreMLModelOptions::default();
    assert!(matches!(opts.compute_platform, coreml_rs::ffi::ComputePlatform::CpuAndGpu));
    assert_eq!(opts.cache_dir, std::path::PathBuf::default());
}

#[test]
fn test_coreml_model_options_custom_cache() {
    let opts = CoreMLModelOptions {
        cache_dir: std::path::PathBuf::from("/tmp/my_cache"),
        ..Default::default()
    };
    assert_eq!(opts.cache_dir.display().to_string(), "/tmp/my_cache");
}

#[test]
fn test_coreml_model_options_cpu_platform() {
    let opts = CoreMLModelOptions {
        compute_platform: coreml_rs::ffi::ComputePlatform::Cpu,
        ..Default::default()
    };
    assert!(matches!(opts.compute_platform, coreml_rs::ffi::ComputePlatform::Cpu));
}
