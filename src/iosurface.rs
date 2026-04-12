//! Thin FFI wrappers for `IOSurface` and `CVPixelBuffer` used by the new
//! zero-copy input API (#828 P0a).
//!
//! This module is macOS-only — the entire `coreml-rs-fork` crate already
//! requires macOS because `build.rs` invokes `swift build`, but the module
//! is additionally gated on `target_os = "macos"` so that the new symbols
//! are not referenced from non-mac targets.
//!
//! The bindings intentionally cover only what the new
//! `add_input_iosurface` / `add_input_cvpixelbuffer_ref` paths need:
//! retain/release of CF types, `IOSurfaceGetAllocSize`, and the lock
//! option constant. Callers are responsible for locking/unlocking the
//! surface around `predict()`. See the P0a design doc for the full
//! lifecycle contract.

#![cfg(target_os = "macos")]
#![allow(non_upper_case_globals)]

use std::ffi::c_void;

/// Opaque CoreFoundation-style pointer to an `IOSurface`.
///
/// This is `IOSurfaceRef` from `<IOSurface/IOSurfaceRef.h>`. It is
/// ABI-compatible with a raw `*const c_void` and can be passed through
/// the swift-bridge FFI boundary.
pub type IOSurfaceRef = *const c_void;

/// Opaque CoreFoundation-style pointer to a `CVPixelBuffer`.
///
/// This is `CVPixelBufferRef` from `<CoreVideo/CVPixelBuffer.h>`.
pub type CVPixelBufferRef = *const c_void;

/// `kIOSurfaceLockReadOnly` — lock flag constant from
/// `<IOSurface/IOSurfaceRef.h>`. Pass this to `IOSurfaceLock` when the
/// caller only needs to read the surface contents (the typical CoreML
/// inference case).
pub const kIOSurfaceLockReadOnly: u32 = 1;

#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    pub fn IOSurfaceGetBaseAddress(surface: IOSurfaceRef) -> *mut c_void;
    pub fn IOSurfaceGetAllocSize(surface: IOSurfaceRef) -> usize;
    pub fn IOSurfaceLock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
    pub fn IOSurfaceUnlock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    pub fn CFRetain(cf: *const c_void) -> *const c_void;
    pub fn CFRelease(cf: *const c_void);
}

/// RAII wrapper around a retained `IOSurfaceRef`.
///
/// Constructing a `RetainedIOSurface` calls `CFRetain` on the passed
/// reference. Dropping it calls `CFRelease`. This mirrors the way
/// CoreFoundation types are handled in `objc2`-style wrappers elsewhere
/// in the workspace, but avoids pulling in a new dependency for this
/// single use.
///
/// # Safety invariant
///
/// The caller of [`RetainedIOSurface::retain`] must pass a non-null
/// pointer that actually points to an `IOSurface` (or a type that
/// inherits from `CFTypeRef`). Passing anything else is undefined
/// behavior.
#[derive(Debug)]
pub struct RetainedIOSurface {
    ptr: IOSurfaceRef,
}

impl RetainedIOSurface {
    /// Retain `surface` and return an owning wrapper.
    ///
    /// # Safety
    ///
    /// `surface` must be a valid `IOSurfaceRef` or null. If null, this
    /// returns an error.
    pub unsafe fn retain(surface: IOSurfaceRef) -> Result<Self, &'static str> {
        if surface.is_null() {
            return Err("IOSurfaceRef is null");
        }
        // SAFETY: caller asserted the pointer is a valid CFTypeRef.
        unsafe { CFRetain(surface) };
        Ok(Self { ptr: surface })
    }

    /// Return the retained pointer without transferring ownership.
    pub fn as_ptr(&self) -> IOSurfaceRef {
        self.ptr
    }
}

impl Drop for RetainedIOSurface {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `ptr` was retained in `retain()` and has not been
            // released anywhere else — this is the single CFRelease that
            // balances the CFRetain.
            unsafe { CFRelease(self.ptr) };
        }
    }
}

// SAFETY: `IOSurfaceRef` values are thread-safe to retain/release and
// read from once locked. `RetainedIOSurface` itself is just an owned
// retain count, which can be moved across threads.
unsafe impl Send for RetainedIOSurface {}
unsafe impl Sync for RetainedIOSurface {}
