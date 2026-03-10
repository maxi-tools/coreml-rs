use std::{path::PathBuf, process::Command};

fn main() {
    let bridge_files = vec!["src/swift.rs"];
    swift_bridge_build::parse_bridges(bridge_files)
        .write_all_concatenated(swift_bridge_out_dir(), "rust-calls-swift");

    // Skip Swift compilation if 'swift' command is not found
    if Command::new("swift").arg("--version").output().is_ok() {
        compile_swift();
    } else {
        println!("cargo:warning=Swift compiler not found. Skipping Swift compilation. Use this mode ONLY for `cargo check --tests`.");
    }

    println!("cargo:rustc-link-lib=static=swift-library");
    println!(
        "cargo:rustc-link-search={}",
        swift_library_static_lib_dir().to_str().unwrap()
    );
}

fn compile_swift() {
    let swift_package_dir = manifest_dir().join("swift-library");
    let triple = std::env::var("TARGET").unwrap();
    let parts = triple.split("-").collect::<Vec<_>>();
    let arch = parts.first().unwrap();

    let mut cmd = Command::new("swift");
    cmd.current_dir(swift_package_dir)
        .arg("build")
        .args(["--arch", arch])
        .args(["-Xswiftc", "-static"])
        .args([
            "-Xswiftc",
            "-import-objc-header",
            "-Xswiftc",
            swift_source_dir()
                .join("bridging-header.h")
                .to_str()
                .unwrap(),
        ]);

    if is_release_build() {
        cmd.args(["-c", "release"]);
    }

    let child = cmd.spawn().unwrap();
    let exit_status = child.wait_with_output().unwrap();
    if !exit_status.status.success() {
        std::process::exit(1);
    }
}

fn swift_bridge_out_dir() -> PathBuf {
    generated_code_dir()
}
fn manifest_dir() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
}
fn is_release_build() -> bool {
    std::env::var("PROFILE").unwrap() == "release"
}
fn swift_source_dir() -> PathBuf {
    manifest_dir().join("swift-library/Sources/swift-library")
}
fn generated_code_dir() -> PathBuf {
    swift_source_dir().join("generated")
}
fn swift_library_static_lib_dir() -> PathBuf {
    let debug_or_release = if is_release_build() {
        "release"
    } else {
        "debug"
    };
    manifest_dir().join(format!("swift-library/.build/{}", debug_or_release))
}
