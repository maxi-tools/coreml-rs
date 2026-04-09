# TODO Tracker

Tracking outstanding work, safety audits, and feature expansion for coreml-rs.

## Completed (Recent)

- [x] ANE output backing use-after-free fix (commit b26ed6b)
- [x] Buffer capacity leak fix (commit 8974c62)
- [x] `input_shape()` API addition for runtime shape queries (commit 53fed64)
- [x] E2E integration tests (cucumber, max-tools CI)
- [x] Safety audit review feedback (commit 3a2ccd9)
- [x] Batch index validation and error context binding

## In Progress

### Memory Safety (Critical)

- [ ] Complete unsafe code audit in `src/mlmodel.rs` and `src/mlbatchmodel.rs`
- [ ] Add SAFETY comments to all unsafe blocks (currently ~80% documented)
- [ ] Validate Send/Sync bounds on all public types
- [ ] Test model unload/reload sequences under memory pressure

### Input Format Support

- [ ] Image preprocessing pipeline (normalization, resize, color space conversion)
- [ ] Support for MLImageType inputs (vs. current NDArray-only)
- [ ] Multi-dimensional array types (3D, 5D+ beyond current 4D max)
- [ ] Zero-copy input binding for large tensors

### Performance

- [ ] Metal shader integration for on-device preprocessing
- [ ] Output buffer pooling to reduce allocation churn
- [ ] Caching strategy for repeated shape queries
- [ ] Benchmark ANE vs CPU vs GPU across model sizes

### Testing

- [ ] Add property-based tests for input shape queries
- [ ] Edge case coverage: empty inputs, mismatched shapes, overflow scenarios
- [ ] Stress test: 1000+ rapid predict cycles
- [ ] Memory profiling on long-running processes

## Backlog

### Advanced Configuration

- [ ] Custom caching policies (TTL, size limits)
- [ ] Priority/QoS settings for concurrent inference
- [ ] Model compilation optimization levels
- [ ] ANE/GPU/CPU affinity control

### Error Handling

- [ ] Graceful degradation (ANE unavailable → GPU fallback)
- [ ] Detailed error codes and recovery suggestions
- [ ] Timeout and cancellation support
- [ ] Model validation diagnostics

### Documentation

- [ ] Architecture.md with design decisions
- [ ] Safety guarantees document
- [ ] Performance tuning guide
- [ ] Troubleshooting guide for common issues

## Known Limitations

1. **Single Threading**: Models are not thread-safe internally; use Arc<Mutex<>> for shared access
2. **Output Format**: Fixed to MLArray dictionary format; no tensor protocol support
3. **ANE Availability**: Graceful fallback to CPU not yet implemented
4. **Model Size**: Tested up to 512MB; >1GB untested
5. **Input Preprocessing**: Currently manual (no built-in normalization)

## Metrics & Health

| Metric | Current | Target |
|--------|---------|--------|
| Unsafe Blocks Documented | 80% | 100% |
| Test Coverage | ~70% | >85% |
| E2E Scenarios | 12 | 20+ |
| Safety Audit Age | 2 weeks | <1 week |

## References

- GitHub Issues: maxi-tools/coreml-rs (PRs #8-18)
- CI: Self-hosted runner (macOS + iOS simulator)
- Tracking: Maxicumber integration test scenarios
