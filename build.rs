use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

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

    // Everything this script produces lives under OUT_DIR. It used to write
    // the generated bridge sources and the SwiftPM `.build` tree into the
    // crate's own source directory, which for a git dependency is the shared
    // `~/.cargo/git/checkouts/coreml-rs-*/<rev>/` tree. Two consequences bit
    // us in 2026-09: a second cargo (another target dir on the same machine,
    // or a fleet CI runner sharing CARGO_HOME) that re-validated that
    // checkout wiped the untracked `.build` and `generated` directories
    // between this script finishing and the final link, so rustc reported
    // "could not find native static library `swift-library`" after a build
    // log that said "Build complete!"; and debug and release builds of the
    // same rev raced each other for the same `.build` directory. Staging a
    // copy of the (tiny) package into OUT_DIR makes the build script pure in
    // the way cargo expects: nothing outside OUT_DIR is written, and every
    // profile/target-dir gets its own scratch tree.
    let staged_package = stage_swift_package();

    // 1. Use `swift-bridge-build` to generate Swift/C FFI glue.
    //    You can also use the `swift-bridge` CLI.
    let bridge_files = vec!["src/swift.rs"];
    let generated_dir = staged_package.join("Sources/swift-library/generated");
    swift_bridge_build::parse_bridges(bridge_files)
        .write_all_concatenated(&generated_dir, "rust-calls-swift");
    export_cdecl_entry_points(&generated_dir);

    // 2. Compile Swift library.
    //
    // The Swift compiler is required — without it the FFI layer cannot be
    // built. `COREML_RS_SKIP_SWIFT=1` is the sanctioned escape hatch for
    // check-only workflows and is required by the project guidelines; an
    // earlier revision of this branch dropped it, which turned a machine
    // without Xcode from "degrades gracefully" into "hard fails".
    let lib_dir = if Command::new("swift").arg("--version").output().is_ok() {
        compile_swift(&staged_package)
    } else if std::env::var("COREML_RS_SKIP_SWIFT").as_deref() == Ok("1") {
        println!("cargo:warning=Swift compiler not found. Skipping Swift compilation (COREML_RS_SKIP_SWIFT=1).");
        return;
    } else {
        panic!("Swift compiler not found. Install Xcode or set COREML_RS_SKIP_SWIFT=1 for check-only builds.");
    };

    // 3. Link to Swift library
    println!("cargo:rustc-link-lib=static=swift-library");
    println!("cargo:rustc-link-search={}", lib_dir.display());

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

/// Copy `swift-library/{Package.swift,Sources}` into `OUT_DIR/swift-library`
/// and return that path. The staged `Sources` tree is replaced, not merged,
/// so a file renamed or deleted upstream cannot linger and be compiled twice;
/// SwiftPM's `.build` scratch tree is kept so its incremental build still
/// applies, and `compile_swift` clears any previous archive from it.
fn stage_swift_package() -> PathBuf {
    let src = manifest_dir().join("swift-library");
    let dst = out_dir().join("swift-library");
    fs::create_dir_all(&dst).unwrap();
    copy_file(&src.join("Package.swift"), &dst.join("Package.swift"));
    let _ = fs::remove_dir_all(dst.join("Sources"));
    copy_tree(&src.join("Sources"), &dst.join("Sources"));
    dst
}

fn copy_file(from: &Path, to: &Path) {
    fs::copy(from, to)
        .unwrap_or_else(|e| panic!("copy {} -> {}: {e}", from.display(), to.display()));
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap_or_else(|e| panic!("read {}: {e}", from.display())) {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            // `generated` is a build product, never a source; skip a stale one
            // left by the pre-OUT_DIR layout.
            if entry.file_name() == "generated" {
                continue;
            }
            copy_tree(&entry.path(), &target);
        } else {
            copy_file(&entry.path(), &target);
        }
    }
}

/// Make every `@_cdecl` function swift-bridge generated `public`.
///
/// swift-bridge emits its C entry points as `@_cdecl("...") func ...` with
/// the default `internal` access. Under `-O -whole-module-optimization`
/// (every release build) the compiler gives internal symbols hidden
/// visibility, and Xcode 27's Swift Build system then prelinks the static
/// library's objects with `ld -r`, which turns hidden symbols into plain
/// locals: `nm -m` reports `non-external (was a private external)` for
/// `___swift_bridge__$modelWithPath` and the Rust link fails with
/// "Undefined symbols for architecture arm64" although the archive is
/// present. The native SwiftPM build system (Xcode ≤ 26, deprecated in 27)
/// skipped the prelink, so the hidden symbols stayed linkable and the
/// problem never showed. Entry points meant for a foreign linker are public
/// API by definition; marking them so keeps them external under every
/// combination of optimisation level and build system. Every parameter type
/// they use is a pointer, a scalar, or a C type from the bridging header,
/// all of which Swift treats as public, so the change compiles cleanly.
fn export_cdecl_entry_points(generated_dir: &Path) {
    for file in [
        generated_dir.join("rust-calls-swift/rust-calls-swift.swift"),
        generated_dir.join("SwiftBridgeCore.swift"),
    ] {
        let Ok(source) = fs::read_to_string(&file) else {
            continue;
        };
        let mut out = String::with_capacity(source.len() + 512);
        let mut after_cdecl = false;
        for line in source.lines() {
            if after_cdecl && line.starts_with("func ") {
                out.push_str("public ");
            }
            out.push_str(line);
            out.push('\n');
            after_cdecl = line.trim_start().starts_with("@_cdecl(");
        }
        fs::write(&file, out).unwrap_or_else(|e| panic!("write {}: {e}", file.display()));
    }
}

/// Run `swift build` on the staged package and return the directory that
/// holds `libswift-library.a`.
fn compile_swift(package_dir: &Path) -> PathBuf {
    let triple = std::env::var("TARGET").unwrap();
    // Rust spells Apple silicon `aarch64`; Xcode spells it `arm64`. The old
    // SwiftPM native build system accepted either, but the Swift Build
    // system that Xcode 27 makes the default validates ARCHS and, given
    // `aarch64`, prints "None of the architectures in ARCHS (aarch64) are
    // valid", builds no product and still exits 0 with "Build complete!" --
    // after which rustc fails with "could not find native static library
    // `swift-library`". Every maxi-ml lane on an Xcode 27 runner failed that
    // way from 2026-09-16 until this mapping.
    let arch = match triple.split('-').next().unwrap_or_default() {
        "aarch64" => "arm64",
        other => other,
    };

    let scratch = package_dir.join(".build");
    // Whatever archive a previous run left in the scratch tree — possibly
    // under a different layout, if the toolchain changed — must not be what
    // `find_static_lib` picks up below.
    remove_static_libs(&scratch);
    let mut cmd = Command::new("swift");

    cmd.current_dir(package_dir)
        .arg("build")
        .arg("--scratch-path")
        .arg(&scratch)
        .args(["--arch", arch])
        .args(["-Xswiftc", "-static"])
        .args(["-Xswiftc", "-import-objc-header", "-Xswiftc"])
        .arg(package_dir.join("Sources/swift-library/bridging-header.h"));

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

    // SwiftPM's native build system leaves the product at
    // `.build/<profile>/libswift-library.a` (a symlink into
    // `.build/<triple>/<profile>/`); Xcode 27's Swift Build system puts it
    // under `.build/out/Products/<Profile>/`. Rather than hard-code either,
    // find the archive and link-search wherever it landed. A build that
    // "completed" without producing it is the Xcode 27 ARCHS failure above
    // (or a sibling of it), so name the scratch tree in the panic.
    find_static_lib(&scratch).unwrap_or_else(|| {
        panic!(
            "swift build reported success but produced no libswift-library.a under {}",
            scratch.display()
        )
    })
}

const STATIC_LIB: &str = "libswift-library.a";

/// Every `libswift-library.a` under `dir`, following directories but not
/// symlinks (SwiftPM's `.build/<profile>` symlink would otherwise list the
/// same archive twice).
fn static_libs_under(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            static_libs_under(&path, found);
        } else if file_type.is_file()
            && path.file_name().and_then(|n| n.to_str()) == Some(STATIC_LIB)
        {
            found.push(path);
        }
    }
}

fn remove_static_libs(dir: &Path) {
    let mut stale = Vec::new();
    static_libs_under(dir, &mut stale);
    for path in stale {
        fs::remove_file(&path).unwrap_or_else(|e| panic!("remove stale {}: {e}", path.display()));
    }
}

fn find_static_lib(dir: &Path) -> Option<PathBuf> {
    let mut found = Vec::new();
    static_libs_under(dir, &mut found);
    // The archive's own directory, not the profile symlink, so the link
    // search path is stable across SwiftPM layouts.
    found
        .first()
        .and_then(|p| p.parent().map(Path::to_path_buf))
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
}

fn out_dir() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").unwrap())
}

fn is_release_build() -> bool {
    std::env::var("PROFILE").unwrap() == "release"
}
