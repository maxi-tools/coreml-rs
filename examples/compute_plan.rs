//! Print the compute-plan device counts for a model.
//!
//! Usage: cargo run --example compute_plan -- <path.mlpackage> [cpu|cpu+ane|cpu+gpu|all]
use coreml_rs_fork::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: compute_plan <model> [units]");
    let platform = match args.next().as_deref() {
        Some("cpu") => ComputePlatform::Cpu,
        Some("cpu+gpu") => ComputePlatform::CpuAndGpu,
        Some("all") => ComputePlatform::All,
        _ => ComputePlatform::CpuAndANE,
    };

    let opts = CoreMLModelOptions::new().with_compute_platform(platform);
    let model = CoreMLModelWithState::new(&path, opts)
        .load()
        .expect("model load failed");
    eprintln!("compiled path: {:?}", model.compiled_path());
    match model.compute_plan_device_counts() {
        Some(counts) => println!(
            "total={} ane={} gpu={} cpu={} (ane fraction {:.1}%)",
            counts.total,
            counts.ane,
            counts.gpu,
            counts.cpu,
            counts.ane_fraction() * 100.0
        ),
        None => println!("no compute plan available"),
    }
}
