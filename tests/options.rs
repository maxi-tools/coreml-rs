use coreml_rs::CoreMLModelOptions;

#[test]
fn test_coreml_model_options_default() {
    let opts = CoreMLModelOptions::default();
    assert_eq!(opts.compute_platform, coreml_rs::ffi::ComputePlatform::All);
    assert_eq!(opts.cache_dir.to_str().unwrap(), ".");
}

#[test]
fn test_coreml_model_options_custom_cache() {
    let mut opts = CoreMLModelOptions::default();
    opts.cache_dir = std::path::PathBuf::from("/tmp/my_cache");
    assert_eq!(opts.cache_dir.to_str().unwrap(), "/tmp/my_cache");
}

#[test]
fn test_coreml_model_options_cpu_platform() {
    let mut opts = CoreMLModelOptions::default();
    opts.compute_platform = coreml_rs::ffi::ComputePlatform::CpuOnly;
    assert_eq!(opts.compute_platform, coreml_rs::ffi::ComputePlatform::CpuOnly);
}
