use std::{
    cell::UnsafeCell,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use coreml_rs_fork::{
    ComputePlatform, CoreMLModelOptions, CoreMLModelWithState, PredictRetryOptions,
};
use ndarray::{ArrayD, IxDyn};

#[test]
#[ignore = "load-heavy local CoreML stress test; requires target/vitae.mlpackage"]
fn retry_vitae_under_concurrent_load() {
    let model_paths = stress_model_paths();
    if model_paths.is_empty() {
        eprintln!("skipping: no local stress models found under target/*.mlpackage");
        return;
    }

    let workers = env_usize("COREML_RETRY_STRESS_WORKERS").unwrap_or(1).max(1);
    let total_iterations = env_usize("COREML_RETRY_STRESS_ITERATIONS").unwrap_or(1000);
    let max_retries = env_usize("COREML_RETRY_STRESS_MAX_RETRIES").unwrap_or(3);
    let progress_every = env_usize("COREML_RETRY_STRESS_PROGRESS_EVERY").unwrap_or(100);
    let compute_platform =
        env_compute_platform("COREML_RETRY_STRESS_COMPUTE").unwrap_or(ComputePlatform::CpuAndANE);
    let disable_experimental_mle =
        env_bool("COREML_RETRY_STRESS_DISABLE_EXPERIMENTAL_MLE").unwrap_or(false);
    let run_retry = env_bool("COREML_RETRY_STRESS_RUN_RETRY").unwrap_or(true);

    let require_baseline_failure = std::env::var_os("COREML_RETRY_STRESS_REQUIRE_BASELINE_FAILURE")
        .is_some_and(|value| value != "0");

    // Everything except which pass this is is identical between the two,
    // so it is built once and shared.
    let config = StressConfig {
        model_paths,
        workers,
        total_iterations,
        progress_every,
        compute_platform,
        disable_experimental_mle,
    };

    let baseline = stress_predict(
        &config,
        &StressRun {
            label: "baseline",
            // The baseline must not retry, or there is nothing to compare
            // the retry pass against.
            retry_options: None,
        },
    );
    let retry = run_retry.then(|| {
        stress_predict(
            &config,
            &StressRun {
                label: "retry",
                retry_options: Some(PredictRetryOptions::fixed(
                    max_retries,
                    Duration::from_millis(10),
                )),
            },
        )
    });

    // Names, not just a count: when a run fails you want to know which
    // model. Borrowed from the config, which owns the paths.
    let model_names = config
        .model_paths
        .iter()
        .map(|path| path.file_name().unwrap().to_string_lossy())
        .collect::<Vec<_>>()
        .join(",");

    eprintln!(
        "coreml stress: models={model_names}, workers={workers}, total_iterations={total_iterations}, compute={}, disable_experimental_mle={disable_experimental_mle}, baseline_failures={}, retry_failures={}, retries_attempted={}",
        compute_platform_name(compute_platform),
        baseline.failures,
        retry
            .as_ref()
            .map(|result| result.failures.to_string())
            .unwrap_or_else(|| "skipped".to_string()),
        retry
            .as_ref()
            .map(|result| result.retries.to_string())
            .unwrap_or_else(|| "skipped".to_string())
    );
    baseline
        .timings
        .print("baseline", baseline.completed, baseline.predictions);
    if let Some(retry) = &retry {
        retry
            .timings
            .print("retry", retry.completed, retry.predictions);
    }
    print_samples("baseline", &baseline.sample_errors);
    if let Some(retry) = &retry {
        print_samples("retry", &retry.sample_errors);
    }

    if require_baseline_failure {
        assert!(
            baseline.failures > 0,
            "baseline run did not reproduce prediction failures"
        );
    }

    if let Some(retry) = retry {
        // Guard against a vacuous comparison. With a clean baseline the
        // assertion below is `0 <= 0`, which passes whether or not the
        // retry logic works at all -- so a retry run requires the
        // baseline to have actually failed.
        assert!(
            baseline.failures > 0,
            "baseline run did not reproduce prediction failures; the retry \
             comparison would be vacuous (set \
             COREML_RETRY_STRESS_REQUIRE_BASELINE_FAILURE=0 to accept that)"
        );

        // And the retry path must have run. Without this, a regression
        // that set max_retries to 0 -- or that dropped the per-attempt
        // re-bind -- would still pass the failure-count comparison.
        assert!(
            retry.retries > 0,
            "no retries were attempted, so predict_with_rebind_retry_if never \
             reached a second attempt (max_retries={max_retries})"
        );
        assert!(
            retry.failures <= baseline.failures,
            "retry should not increase final prediction failures: baseline={} retry={}",
            baseline.failures,
            retry.failures
        );
        eprintln!(
            "coreml stress: {} retries attempted, baseline_failures={}, retry_failures={}",
            retry.retries, baseline.failures, retry.failures
        );
    }
}

struct StressResult {
    completed: usize,
    predictions: usize,
    failures: usize,
    /// Number of retry attempts actually made. Zero means the retry path
    /// never ran, which makes any failure-count comparison vacuous.
    retries: usize,
    sample_errors: Vec<String>,
    timings: StressTimings,
}

/// What to run: the same for every stress pass, differing only in
/// [`StressRun::label`] and [`StressRun::retry_options`].
struct StressConfig {
    model_paths: Vec<PathBuf>,
    workers: usize,
    total_iterations: usize,
    progress_every: usize,
    compute_platform: ComputePlatform,
    disable_experimental_mle: bool,
}

/// One stress pass.
struct StressRun {
    /// Distinguishes this pass in the log line.
    label: &'static str,
    /// `None` for the baseline pass -- which must not retry, so that the
    /// retry pass has something to be compared against. `Some(..)` enables
    /// the policy for the retry pass.
    retry_options: Option<PredictRetryOptions>,
}

fn stress_predict(config: &StressConfig, run: &StressRun) -> StressResult {
    let model_paths = config.model_paths.as_slice();
    let workers = config.workers;
    let total_iterations = config.total_iterations;
    let progress_every = config.progress_every;
    let compute_platform = config.compute_platform;
    let disable_experimental_mle = config.disable_experimental_mle;
    let label = run.label;
    let retry_options = run.retry_options;
    let start = Instant::now();
    let completed = Arc::new(AtomicUsize::new(0));
    let predictions = Arc::new(AtomicUsize::new(0));
    let failures = Arc::new(AtomicUsize::new(0));
    let sample_errors = Arc::new(Mutex::new(Vec::new()));
    // Counts retries actually attempted, so the test can assert the retry
    // path ran rather than assuming it did.
    let retries = Arc::new(AtomicUsize::new(0));
    let timings = Arc::new(StressTimings::default());
    let mut handles = Vec::with_capacity(workers);

    let models = Arc::new(load_stress_models(
        model_paths,
        compute_platform,
        disable_experimental_mle,
        &timings,
    ));

    for worker_idx in 0..workers {
        let worker_iterations = worker_iterations(total_iterations, workers, worker_idx);
        let models = Arc::clone(&models);
        let completed = Arc::clone(&completed);
        let predictions = Arc::clone(&predictions);
        let failures = Arc::clone(&failures);
        let sample_errors = Arc::clone(&sample_errors);
        let retries = Arc::clone(&retries);
        let timings = Arc::clone(&timings);
        handles.push(thread::spawn(move || {
            let mut rng = XorShift64::new(0x4d595df4d0f33173 ^ worker_idx as u64);

            for _ in 0..worker_iterations {
                for stress_model in models.iter() {
                    let input_start = Instant::now();
                    let input = ArrayD::<f32>::from_shape_fn(
                        IxDyn(stress_model.input_shape.as_slice()),
                        |_| rng.next_f32(),
                    );
                    timings
                        .input_ns
                        .fetch_add(duration_ns(input_start.elapsed()), Ordering::Relaxed);

                    // Bind and predict as one retryable unit.
                    //
                    // A failed predict clears the Swift-side input dict
                    // (`clearBindings()` in the `predict()` catch), so a
                    // retry has to re-install the inputs. `input` is
                    // cloned per attempt rather than moved because the
                    // closure can run more than once.
                    let bind_start = Instant::now();
                    let model = stress_model.model.get();
                    let input_name = stress_model.input_name.clone();
                    let retry_counters = Arc::clone(&retries);
                    let result = match retry_options {
                        Some(options) => model.predict_with_rebind_retry_if(
                            options,
                            move |_err| {
                                retry_counters.fetch_add(1, Ordering::Relaxed);
                                true
                            },
                            |model| {
                                model.add_input(&input_name, input.clone())?;
                                model.predict()
                            },
                        ),
                        None => (|model: &mut CoreMLModelWithState| {
                            // Routed through a closure so this arm has
                            // the same shape as the retry arm and can
                            // use `?`: the worker closure itself returns
                            // `()`, so `?` cannot appear inline here.
                            model.add_input(&input_name, input.clone())?;
                            model.predict()
                        })(model),
                    };
                    timings
                        .bind_ns
                        .fetch_add(duration_ns(bind_start.elapsed()), Ordering::Relaxed);

                    let predict_start = Instant::now();
                    if let Err(err) = result {
                        failures.fetch_add(1, Ordering::Relaxed);
                        let mut sample_errors = sample_errors.lock().unwrap();
                        if sample_errors.len() < 8 {
                            sample_errors.push(format!("{}: {err}", stress_model.name));
                        }
                    }
                    timings
                        .predict_ns
                        .fetch_add(duration_ns(predict_start.elapsed()), Ordering::Relaxed);
                    predictions.fetch_add(1, Ordering::Relaxed);
                }

                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if should_log_progress(progress_every, done, total_iterations) {
                    eprintln!("{label}: completed {done}/{total_iterations} requests");
                }
            }
        }));
    }

    for handle in handles {
        handle.join().expect("stress worker panicked");
    }

    timings
        .wall_ns
        .store(duration_ns(start.elapsed()), Ordering::Relaxed);
    let completed = completed.load(Ordering::Relaxed);
    let predictions = predictions.load(Ordering::Relaxed);
    let failures = failures.load(Ordering::Relaxed);
    let retries = retries.load(Ordering::Relaxed);
    let sample_errors = Arc::try_unwrap(sample_errors)
        .expect("all workers should have released sample error collector")
        .into_inner()
        .expect("sample error collector mutex should not be poisoned");
    let timings = Arc::try_unwrap(timings).expect("all workers should have released timings");

    StressResult {
        completed,
        predictions,
        failures,
        retries,
        sample_errors,
        timings,
    }
}

struct StressModel {
    name: String,
    model: SharedModel,
    input_name: String,
    input_shape: Vec<usize>,
}

fn load_stress_models(
    model_paths: &[PathBuf],
    compute_platform: ComputePlatform,
    disable_experimental_mle: bool,
    timings: &StressTimings,
) -> Vec<StressModel> {
    model_paths
        .iter()
        .map(|model_path| {
            let options = CoreMLModelOptions::default()
                .with_compute_platform(compute_platform)
                .with_disable_experimental_mle(disable_experimental_mle);

            let load_start = Instant::now();
            let model = CoreMLModelWithState::new(model_path, options)
                .load()
                .unwrap_or_else(|err| panic!("failed to load {}: {err}", model_path.display()));
            timings
                .load_ns
                .fetch_add(duration_ns(load_start.elapsed()), Ordering::Relaxed);

            let shapes_start = Instant::now();
            let input_shapes = model
                .input_shapes()
                .unwrap_or_else(|err| panic!("failed to read input shapes: {err}"));
            timings
                .shape_ns
                .fetch_add(duration_ns(shapes_start.elapsed()), Ordering::Relaxed);
            let Some((input_name, input_shape)) = input_shapes.into_iter().next() else {
                panic!("model has no inputs: {}", model_path.display());
            };

            StressModel {
                name: model_path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
                model: SharedModel::new(model),
                input_name,
                input_shape,
            }
        })
        .collect()
}

fn stress_model_paths() -> Vec<PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("target/saliency.mlpackage"),
        manifest_dir.join("target/vitae.mlpackage"),
    ];
    candidates
        .into_iter()
        .filter(|path| path.exists())
        .collect()
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn env_bool(name: &str) -> Option<bool> {
    match std::env::var(name).ok()?.as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn env_compute_platform(name: &str) -> Option<ComputePlatform> {
    match std::env::var(name).ok()?.as_str() {
        "ane" | "cpu-and-ane" | "cpu_and_ane" => Some(ComputePlatform::CpuAndANE),
        "gpu" | "cpu-and-gpu" | "cpu_and_gpu" => Some(ComputePlatform::CpuAndGpu),
        "cpu" | "cpu-only" | "cpu_only" => Some(ComputePlatform::Cpu),
        _ => None,
    }
}

fn compute_platform_name(compute_platform: ComputePlatform) -> &'static str {
    match compute_platform {
        ComputePlatform::Cpu => "cpu",
        ComputePlatform::CpuAndANE => "cpu-and-ane",
        ComputePlatform::CpuAndGpu => "cpu-and-gpu",
        ComputePlatform::All => "all",
    }
}

fn should_log_progress(progress_every: usize, completed: usize, iterations: usize) -> bool {
    completed == iterations || (progress_every > 0 && completed.is_multiple_of(progress_every))
}

fn worker_iterations(total_iterations: usize, workers: usize, worker_idx: usize) -> usize {
    let base = total_iterations / workers;
    let extra = usize::from(worker_idx < total_iterations % workers);
    base + extra
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

#[derive(Default, Debug)]
struct StressTimings {
    wall_ns: AtomicU64,
    load_ns: AtomicU64,
    shape_ns: AtomicU64,
    input_ns: AtomicU64,
    bind_ns: AtomicU64,
    predict_ns: AtomicU64,
}

impl StressTimings {
    fn print(&self, label: &str, completed: usize, predictions: usize) {
        let predictions = predictions.max(1) as f64;
        let wall = Duration::from_nanos(self.wall_ns.load(Ordering::Relaxed));
        let load = Duration::from_nanos(self.load_ns.load(Ordering::Relaxed));
        let shape = Duration::from_nanos(self.shape_ns.load(Ordering::Relaxed));
        let input = Duration::from_nanos(self.input_ns.load(Ordering::Relaxed));
        let bind = Duration::from_nanos(self.bind_ns.load(Ordering::Relaxed));
        let predict = Duration::from_nanos(self.predict_ns.load(Ordering::Relaxed));
        eprintln!(
            "{label} timings: requests={}, predictions={}, wall={:.2?}, load_total={:.2?}, shape_total={:.2?}, input_avg={:.2?}, bind_avg={:.2?}, predict_avg={:.2?}",
            completed,
            predictions as usize,
            wall,
            load,
            shape,
            input.div_f64(predictions),
            bind.div_f64(predictions),
            predict.div_f64(predictions),
        );
    }
}

struct SharedModel {
    model: UnsafeCell<CoreMLModelWithState>,
}

unsafe impl Send for SharedModel {}
unsafe impl Sync for SharedModel {}

impl SharedModel {
    fn new(model: CoreMLModelWithState) -> Self {
        Self {
            model: UnsafeCell::new(model),
        }
    }

    // SAFETY CONTRACT: this is deliberately unsound.
    //
    // `get` hands every worker thread a `&mut` into the same
    // `UnsafeCell<CoreMLModelWithState>`, so two threads can hold aliasing
    // `&mut` simultaneously. That violates Stacked Borrows and is exactly
    // what `clippy::mut_from_ref` denies by default -- correctly, since the
    // helper would be a real bug in any non-test caller.
    //
    // It stays because this is an ignored stress test whose whole purpose is
    // to reproduce the unsynchronized concurrent access the retry path is
    // supposed to survive: it looks for CoreML/ANE failures, data
    // corruption, and crashes under load, not for Rust-level safety. Making
    // the borrow sound (one `Mutex` around the model) would serialize the
    // threads and destroy the contention the test is measuring.
    //
    // The allow is scoped to this one function rather than the crate so that
    // the lint stays armed everywhere else in the test.
    #[allow(clippy::mut_from_ref)]
    fn get(&self) -> &mut CoreMLModelWithState {
        // This ignored stress test intentionally shares one loaded model across
        // threads without an external lock to reproduce the production access
        // pattern that relies on lower-level synchronization.
        unsafe { &mut *self.model.get() }
    }
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x as u32) as f32 / u32::MAX as f32
    }
}

fn print_samples(label: &str, sample_errors: &[String]) {
    if sample_errors.is_empty() {
        eprintln!("{label} sample errors: none");
        return;
    }

    eprintln!("{label} sample errors:");
    for err in sample_errors {
        eprintln!("  {err}");
    }
}
