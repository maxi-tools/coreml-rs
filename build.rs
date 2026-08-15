use std::{path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/swift.rs");
    println!("cargo:rerun-if-changed=swift-library/Sources/swift-library");
    println!("cargo:rerun-if-changed=swift-library/Package.swift");

    // Apple targets this crate actually supports.
    //
    // This deliberately does NOT include watchos/tvos. The bundled Swift
    // package declares `platforms: [.macOS(.v13)]` and nothing else, so there
    // is no watchOS/tvOS Swift product to link even if this guard let those
    // targets through — widening it would swap a clean "no Swift bridge built"
    // for a confusing Swift build failure. The Rust side agrees: `iosurface`
    // and large parts of `mlmodel`/`mlarray` are gated on
    // `target_os = "macos"`, and the README's requirements section lists only
    // macOS (build) and iOS (deployment).
    //
    // A stray sentence in the README used to claim watchOS/tvOS support; that
    // claim was never backed by code and has been corrected rather than
    // papered over here. If watchOS/tvOS are ever genuinely targeted, the
    // change belongs in Package.swift and the cfg gates first, and only then
    // in this list.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if !matches!(target_os.as_str(), "macos" | "ios") {
        return;
    }

    // 1. Use `swift-bridge-build` to generate Swift/C FFI glue.
    //    You can also use the `swift-bridge` CLI.
    let bridge_files = vec!["src/swift.rs"];
    swift_bridge_build::parse_bridges(bridge_files)
        .write_all_concatenated(swift_bridge_out_dir(), "rust-calls-swift");

    // 2. Compile Swift library.
    //
    // The Swift compiler is required — without it the FFI layer cannot be
    // built. `COREML_RS_SKIP_SWIFT=1` is the sanctioned escape hatch for
    // check-only workflows and is required by the project guidelines; an
    // earlier revision of this branch dropped it, which turned a machine
    // without Xcode from "degrades gracefully" into "hard fails".
    if Command::new("swift").arg("--version").output().is_ok() {
        compile_swift();
    } else if std::env::var("COREML_RS_SKIP_SWIFT").as_deref() == Ok("1") {
        println!("cargo:warning=Swift compiler not found. Skipping Swift compilation (COREML_RS_SKIP_SWIFT=1).");
        return;
    } else {
        panic!("Swift compiler not found. Install Xcode or set COREML_RS_SKIP_SWIFT=1 for check-only builds.");
    }

    // 3. Link to Swift library
    println!("cargo:rustc-link-lib=static=swift-library");
    println!(
        "cargo:rustc-link-search={}",
        swift_library_static_lib_dir().to_str().unwrap()
    );

    // Without this we will get warnings about not being able to find dynamic libraries, and then
    // we won't be able to compile since the Swift static libraries depend on them:
    // For example:
    // ld: warning: Could not find or use auto-linked library 'swiftCompatibility51'
    // ld: warning: Could not find or use auto-linked library 'swiftCompatibility50'
    // ld: warning: Could not find or use auto-linked library 'swiftCompatibilityDynamicReplacements'
    // ld: warning: Could not find or use auto-linked library 'swiftCompatibilityConcurrency'
    let xcode_path = if let Ok(output) = std::process::Command::new("xcode-select")
        .arg("--print-path")
        .output()
    {
        String::from_utf8(output.stdout.as_slice().into())
            .unwrap()
            .trim()
            .to_string()
    } else {
        "/Applications/Xcode.app/Contents/Developer".to_string()
    };
    println!(
        "cargo:rustc-link-search={}/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx/",
        xcode_path
    );
    println!("cargo:rustc-link-search=/usr/lib/swift");
    // Runtime rpath for the Swift runtime dylibs (libswift_Concurrency etc.)
    // so this crate's own test/example binaries can launch without every consumer
    // needing a rustflags rpath workaround.
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
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

    let child = cmd.spawn().unwrap_or_else(|e| {
        eprintln!("Failed to spawn swift build command: {}", e);
        std::process::exit(1);
    });
    let exit_status = child.wait_with_output().unwrap_or_else(|e| {
        eprintln!("Failed to wait for swift build: {}", e);
        std::process::exit(1);
    });

    if !exit_status.status.success() {
        eprintln!(
            "Swift build failed:\nStderr: {}\nStdout: {}",
            String::from_utf8_lossy(&exit_status.stderr),
            String::from_utf8_lossy(&exit_status.stdout),
        );
        std::process::exit(1);
    }
}

fn swift_bridge_out_dir() -> PathBuf {
    generated_code_dir()
}

fn manifest_dir() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(manifest_dir)
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
