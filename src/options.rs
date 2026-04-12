use crate::ffi::ComputePlatform;
use std::path::PathBuf;

#[derive(Clone)]
pub struct CoreMLModelOptions {
    pub compute_platform: ComputePlatform,
    pub cache_dir: PathBuf,
    /// Maps to MLModelConfiguration.allowLowPrecisionAccumulationOnGPU when set.
    pub allow_low_precision_accumulation_on_gpu: Option<bool>,
    /// Maps to MLPredictionOptions.usesCPUOnly when set.
    pub prediction_uses_cpu_only: Option<bool>,
}

impl Default for CoreMLModelOptions {
    fn default() -> Self {
        Self {
            compute_platform: ComputePlatform::CpuAndANE,
            cache_dir: PathBuf::new(),
            allow_low_precision_accumulation_on_gpu: None,
            prediction_uses_cpu_only: None,
        }
    }
}

impl CoreMLModelOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_compute_platform(mut self, platform: ComputePlatform) -> Self {
        self.compute_platform = platform;
        self
    }

    pub fn with_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = path.into();
        self
    }

    pub fn with_allow_low_precision_accumulation_on_gpu(mut self, enabled: bool) -> Self {
        self.allow_low_precision_accumulation_on_gpu = Some(enabled);
        self
    }

    pub fn with_prediction_uses_cpu_only(mut self, enabled: bool) -> Self {
        self.prediction_uses_cpu_only = Some(enabled);
        self
    }
}

impl std::fmt::Debug for CoreMLModelOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreMLModelOptions")
            .field(
                "compute_platform",
                match self.compute_platform {
                    ComputePlatform::Cpu => &"CPU",
                    ComputePlatform::CpuAndANE => &"CpuAndAne",
                    ComputePlatform::CpuAndGpu => &"CpuAndGpu",
                },
            )
            .field(
                "allow_low_precision_accumulation_on_gpu",
                &self.allow_low_precision_accumulation_on_gpu,
            )
            .field("prediction_uses_cpu_only", &self.prediction_uses_cpu_only)
            .finish()
    }
}

// Info required to create a coreml model
#[derive(Debug, Clone)]
pub struct CoreMLModelInfo {
    pub opts: CoreMLModelOptions,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_ml_model_options_debug() {
        let opts = CoreMLModelOptions::new()
            .with_compute_platform(ComputePlatform::Cpu)
            .with_allow_low_precision_accumulation_on_gpu(true)
            .with_prediction_uses_cpu_only(false);

        let debug_str = format!("{:?}", opts);

        assert!(debug_str.contains("CoreMLModelOptions"));
        assert!(debug_str.contains("compute_platform: \"CPU\""));
        assert!(debug_str.contains("allow_low_precision_accumulation_on_gpu: Some(true)"));
        assert!(debug_str.contains("prediction_uses_cpu_only: Some(false)"));

        let opts2 = CoreMLModelOptions::new().with_compute_platform(ComputePlatform::CpuAndANE);
        let debug_str2 = format!("{:?}", opts2);
        assert!(debug_str2.contains("compute_platform: \"CpuAndAne\""));

        let opts3 = CoreMLModelOptions::new().with_compute_platform(ComputePlatform::CpuAndGpu);
        let debug_str3 = format!("{:?}", opts3);
        assert!(debug_str3.contains("compute_platform: \"CpuAndGpu\""));
    }
}
