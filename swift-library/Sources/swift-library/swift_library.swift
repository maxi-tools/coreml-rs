import CoreML
import CoreVideo
import IOSurface

class BatchOutput {
	var batchProvider: MLBatchProvider? = nil
	var error: String? = nil
	init(error: String? = nil, batchProvider: MLBatchProvider? = nil) {
		self.batchProvider = batchProvider
		self.error = error
	}

	func getOutputAtIndex(at: Int) -> ModelOutput {
		let features = self.batchProvider?.features(at: at) as? MLDictionaryFeatureProvider
		return ModelOutput.init(output: features?.dictionary, cpy: true)
	}

	func count() -> Int {
		let c = self.batchProvider?.count
		guard let c else { return 0 }
		return c
	}

	func getError() -> RustString? {
		if self.error == nil {
			return nil
		}
		return "\(self.error!)".intoRustString()
	}
}

class BatchModelInput {
	var dict: [String: Any] = [:]
	func toFeatureProvider() -> MLDictionaryFeatureProvider? {
		do {
			return try MLDictionaryFeatureProvider.init(dictionary: self.dict)
		} catch {
			return nil
		}
	}
}

class BatchModel: @unchecked Sendable {
	var compiledPath: URL? = nil
	var model: MLModel? = nil
	var modelCompiledAsset: MLModelAsset? = nil
	var inputs: [BatchModelInput] = []
	var computeUnits: MLComputeUnits = .cpuAndNeuralEngine
	var allowLowPrecisionAccumulationOnGPU: Bool? = nil
	var predictionUsesCPUOnly: Bool? = nil
	var failedToLoad: Bool

	init(failedToLoad: Bool = false, model: MLModel? = nil) {
		self.failedToLoad = failedToLoad
	}

	func hasFailedToLoad() -> Bool {
		return self.failedToLoad
	}

	func setAllowLowPrecisionAccumulationOnGPU(enabled: Bool) {
		self.allowLowPrecisionAccumulationOnGPU = enabled
	}

	func setPredictionUsesCPUOnly(enabled: Bool) {
		self.predictionUsesCPUOnly = enabled
	}

	func description() -> ModelDescription {
		return ModelDescription(desc: self.model?.modelDescription)
	}

	func load() -> Bool {
		if hasFailedToLoad() { return false }
		let config = MLModelConfiguration.init()
		config.computeUnits = self.computeUnits
		if let enabled = self.allowLowPrecisionAccumulationOnGPU {
			config.allowLowPrecisionAccumulationOnGPU = enabled
		}
		do {
			if self.compiledPath == nil {
				let semaphore = DispatchSemaphore(value: 0)
				Task { [weak self] in
					guard let self else { return }
					let asset = self.modelCompiledAsset!
					let res = try await MLModel.load(asset: asset, configuration: config)
					self.model = res
					semaphore.signal()
				}
				semaphore.wait()
			} else {
				let loadedModel = try MLModel(contentsOf: self.compiledPath!, configuration: config)
				self.model = loadedModel
			}
			return true
		} catch {
			return false
		}
	}

	func unload() -> Bool {
		if hasFailedToLoad() { return false }
		self.model = nil
		return true
	}

	func bindInputF32(
		shape: RustVec<UInt>, featureName: RustStr, data: UnsafeMutablePointer<Float32>,
		len: UInt, idx: Int
	) -> Bool {
		do {
			var arr: [NSNumber] = []
			var stride: [NSNumber] = []
			var m: UInt = 1
			for i in shape.reversed() {
				stride.append(NSNumber(value: m))
				m = i * m
			}
			stride.reverse()
			for s in shape {
				arr.append(NSNumber(value: s))
			}
			let deallocMultiArrayRust = { (_ ptr: UnsafeMutableRawPointer) in
				rust_vec_free_f32(ptr.assumingMemoryBound(to: Float32.self), len)
			}
			let array = try MLMultiArray.init(
				dataPointer: data, shape: arr, dataType: MLMultiArrayDataType.float32,
				strides: stride, deallocator: deallocMultiArrayRust)
			let value = MLFeatureValue(multiArray: array)
			if self.inputs.count <= idx {
				self.inputs.append(BatchModelInput.init())
			}
			self.inputs[idx].dict[featureName.toString()] = value
			return true
		} catch {
			print("Unexpected input error; \(error)")
			return false
		}
	}

	func predict() -> BatchOutput {
		do {
			let opts = MLPredictionOptions.init()
			if let usesCPUOnly = self.predictionUsesCPUOnly {
				opts.usesCPUOnly = usesCPUOnly
			}
			// TODO (SA): to feature provider
			let features = inputs.compactMap { input in
				input.toFeatureProvider()
			}
			let batchProvider = MLArrayBatchProvider.init(array: features)
			let output = try self.model?.predictions(from: batchProvider, options: opts)
			guard let output else {
				return BatchOutput.init(error: "ran predict without a model loaded into memory")
			}
			return BatchOutput.init(batchProvider: output)
		} catch {
			return BatchOutput.init(error: error.localizedDescription)
		}
	}
}

class ModelDescription {
	var description: MLModelDescription? = nil
	init(desc: MLModelDescription?) {
		self.description = desc
	}

	func failedToLoad() -> Bool { return self.description == nil }

	func inputs() -> RustVec<RustString> {
		let ret = RustVec<RustString>()
		if !failedToLoad() {
			for (_, value) in self.description!.inputDescriptionsByName {
				let str = "\(value)".intoRustString()
				ret.push(value: str)
			}
		}
		return ret
	}
	func outputs() -> RustVec<RustString> {
		let ret = RustVec<RustString>()
		if !failedToLoad() {
			for (_, value) in self.description!.outputDescriptionsByName {
				let str = "\(value)".intoRustString()
				ret.push(value: str)
			}
		}
		return ret
	}
	func output_type(name: RustStr) -> RustString {
		if !failedToLoad() {
			let res = self.description!.outputDescriptionsByName[name.toString()]!
			if res.multiArrayConstraint!.dataType == MLMultiArrayDataType.float32 {
				return "f32".intoRustString()
			}
			if res.multiArrayConstraint!.dataType == MLMultiArrayDataType.float16 {
				return "f16".intoRustString()
			}
		}
		return "".intoRustString()
	}
	func output_shape(name: RustStr) -> RustVec<UInt> {
		if !failedToLoad() {
			let res = self.description?.outputDescriptionsByName[name.toString()]
			guard let res else { return RustVec.init() }
			let arr = res.multiArrayConstraint
			guard let arr else { return RustVec.init() }
			let ret = RustVec<UInt>()
			// Check for flexible dimensions via shapeConstraint range (same as input_shape)
			let ranges = arr.shapeConstraint.sizeRangeForDimension
			let shape = arr.shape
			for i in 0..<shape.count {
				if i < ranges.count {
					let nsRange = ranges[i].rangeValue
					if nsRange.length > 0 {
						ret.push(value: 0)
						continue
					}
				}
				ret.push(value: UInt(truncating: shape[i]))
			}
			return ret
		}
		return RustVec.init()
	}
	func input_shape(name: RustStr) -> RustVec<UInt> {
		if !failedToLoad() {
			let res = self.description?.inputDescriptionsByName[name.toString()]
			guard let res else { return RustVec.init() }
			let arr = res.multiArrayConstraint
			guard let arr else { return RustVec.init() }
			let ret = RustVec<UInt>()
			// Check for flexible dimensions via shapeConstraint range
			let ranges = arr.shapeConstraint.sizeRangeForDimension
			let shape = arr.shape
			for i in 0..<shape.count {
				if i < ranges.count {
					let nsRange = ranges[i].rangeValue  // NSRange: location=min, length=max-min
					if nsRange.length > 0 {
						// Flexible dim — return 0 as wildcard
						ret.push(value: 0)
						continue
					}
				}
				ret.push(value: UInt(truncating: shape[i]))
			}
			return ret
		}
		return RustVec.init()
	}

	func output_names() -> RustVec<RustString> {
		if !failedToLoad() {
			let ret = RustVec<RustString>()
			for (key, _) in self.description!.outputDescriptionsByName {
				ret.push(value: key.intoRustString())
			}
			return ret
		}
		return RustVec.init()
	}

	func input_names() -> RustVec<RustString> {
		if !failedToLoad() {
			let ret = RustVec<RustString>()
			for (key, _) in self.description!.inputDescriptionsByName {
				ret.push(value: key.intoRustString())
			}
			return ret
		}
		return RustVec.init()
	}
}

class ModelOutput {
	var output: [String: Any]? = [:]
	var error: (any Error)? = nil
	var cpy: Bool = false
	init(output: [String: Any]?, error: (any Error)? = nil, cpy: Bool = false) {
		self.output = output
		self.error = error
		self.cpy = cpy
	}
	func hasFailedToLoad() -> Bool {
		return self.error != nil
	}
	func getError() -> RustString? {
		if self.error == nil {
			return nil
		}
		return "\(self.error!)".intoRustString()
	}
	func outputDescription() -> RustVec<RustString> {
		if hasFailedToLoad() { return RustVec.init() }
		let output = self.output!
		let ret = RustVec<RustString>()
		for key in output.keys {
			let str = "\(key):\(output[key]!)".intoRustString()
			ret.push(value: str)
		}
		return ret
	}
	func outputShape(name: RustStr) -> RustVec<UInt> {
		if hasFailedToLoad() { return RustVec.init() }
		guard let out = multiArray(name: name) else { return RustVec.init() }
		let ret = RustVec<UInt>()
		for dim in out.shape {
			ret.push(value: UInt(truncating: dim))
		}
		return ret
	}

	private func multiArray(name: RustStr) -> MLMultiArray? {
		guard let output = self.output, let value = output[name.toString()] else { return nil }
		if let feature = value as? MLFeatureValue {
			return feature.multiArrayValue
		}
		return value as? MLMultiArray
	}

	private func contiguousLayout(for out: MLMultiArray) -> (
		shape: [Int], strides: [Int], expectedStrides: [Int], isContiguous: Bool
	) {
		let shape = out.shape.map { $0.intValue }
		let strides = out.strides.map { $0.intValue }
		var expectedStrides = [Int](repeating: 1, count: shape.count)
		if shape.count > 1 {
			for i in stride(from: shape.count - 2, through: 0, by: -1) {
				expectedStrides[i] = expectedStrides[i + 1] * shape[i + 1]
			}
		}
		return (shape, strides, expectedStrides, strides == expectedStrides)
	}

	/// Traverse a non-contiguous MLMultiArray in strided order, calling `emit` for each element's pointer offset.
	private func stridedTraversal(
		out: MLMultiArray, layout: (shape: [Int], strides: [Int], expectedStrides: [Int], isContiguous: Bool),
		emit: (Int) -> Void
	) {
		let shape = layout.shape
		let strides = layout.strides
		let l = out.count
		var coords = [Int](repeating: 0, count: shape.count)
		var offset = 0
		for _ in 0..<l {
			emit(offset)
			for d in Swift.stride(from: shape.count - 1, through: 0, by: -1) {
				coords[d] += 1
				offset += strides[d]
				if coords[d] < shape[d] { break }
				coords[d] = 0
				offset -= shape[d] * strides[d]
			}
		}
	}

	private func subscriptTraversal(
		out: MLMultiArray, shape: [Int],
		emitSubscript: ([NSNumber]) -> Void
	) {
		let l = out.count
		var coords = [Int](repeating: 0, count: shape.count)
		var indices = [NSNumber](repeating: 0, count: shape.count)
		for _ in 0..<l {
			emitSubscript(indices)
			for d in Swift.stride(from: shape.count - 1, through: 0, by: -1) {
				coords[d] += 1
				if coords[d] < shape[d] {
					indices[d] = NSNumber(value: coords[d])
					break
				}
				coords[d] = 0
				indices[d] = 0
			}
		}
	}

	func outputF32(name: RustStr) -> RustVec<Float32> {
		if hasFailedToLoad() { return RustVec.init() }
		guard let out = multiArray(name: name) else { return RustVec.init() }

		let l = out.count
		let layout = contiguousLayout(for: out)
		if !layout.isContiguous {
			print("[STRIDE BUG] outputF32 \(name.toString()): shape=\(layout.shape) strides=\(layout.strides) expected=\(layout.expectedStrides) count=\(l)")
		}

		if layout.isContiguous && out.dataType == .float32 {
			let ptr = out.dataPointer.assumingMemoryBound(to: Float32.self)
			return self.cpy ? rust_vec_from_ptr_f32_cpy(ptr, UInt(l)) : rust_vec_from_ptr_f32(ptr, UInt(l))
		}

		var v = RustVec<Float32>()
		if out.dataType == .float32 {
			let ptr = out.dataPointer.assumingMemoryBound(to: Float32.self)
			stridedTraversal(out: out, layout: layout, emit: { v.push(value: ptr[$0]) })
		} else {
			subscriptTraversal(out: out, shape: layout.shape, emitSubscript: { v.push(value: out[$0].floatValue) })
		}
		return v
	}

	func outputI32(name: RustStr) -> RustVec<Int32> {
		if hasFailedToLoad() { return RustVec.init() }
		guard let out = multiArray(name: name) else { return RustVec.init() }

		let l = out.count
		let layout = contiguousLayout(for: out)

		if layout.isContiguous && out.dataType == .int32 {
			let ptr = out.dataPointer.assumingMemoryBound(to: Int32.self)
			return self.cpy ? rust_vec_from_ptr_i32_cpy(ptr, UInt(l)) : rust_vec_from_ptr_i32(ptr, UInt(l))
		}

		var v = RustVec<Int32>()
		if out.dataType == .int32 {
			let ptr = out.dataPointer.assumingMemoryBound(to: Int32.self)
			stridedTraversal(out: out, layout: layout, emit: { v.push(value: ptr[$0]) })
		} else {
			subscriptTraversal(out: out, shape: layout.shape, emitSubscript: { v.push(value: out[$0].int32Value) })
		}
		return v
	}

	func outputU16(name: RustStr) -> RustVec<UInt16> {
		if hasFailedToLoad() { return RustVec.init() }
		guard let out = multiArray(name: name) else { return RustVec.init() }

		let l = out.count
		let layout = contiguousLayout(for: out)
		if !layout.isContiguous {
			print("[STRIDE BUG] outputU16 \(name.toString()): shape=\(layout.shape) strides=\(layout.strides) expected=\(layout.expectedStrides) count=\(l)")
		}

		if layout.isContiguous && out.dataType == .float16 {
			let ptr = out.dataPointer.assumingMemoryBound(to: UInt16.self)
			return self.cpy ? rust_vec_from_ptr_u16_cpy(ptr, UInt(l)) : rust_vec_from_ptr_u16(ptr, UInt(l))
		}

		// For f16 data, subscript returns NSNumber wrapping a float —
		// .uint16Value would truncate to integer, so we round-trip through
		// Float16 to preserve the raw bit pattern.
		var v = RustVec<UInt16>()
		if out.dataType == .float16 {
			let ptr = out.dataPointer.assumingMemoryBound(to: UInt16.self)
			stridedTraversal(out: out, layout: layout, emit: { v.push(value: ptr[$0]) })
		} else {
			subscriptTraversal(out: out, shape: layout.shape, emitSubscript: {
				let f16val = Float16(out[$0].floatValue)
				v.push(value: f16val.bitPattern)
			})
		}
		return v
	}
}

func initWithCompiledAsset(
	ptr: UnsafeMutablePointer<UInt8>, len: Int, compute: ComputePlatform
) -> Model {
	var computeUnits: MLComputeUnits
	switch compute {
	case .Cpu:
		computeUnits = .cpuOnly
		break
	case .CpuAndANE:
		computeUnits = .cpuAndNeuralEngine
		break
	case .CpuAndGpu:
		computeUnits = .cpuAndGPU
		break
	}
	let data = Data.init(
		bytesNoCopy: ptr, count: len,
		deallocator: Data.Deallocator.custom { ptr, len in
			rust_vec_free_u8(ptr.assumingMemoryBound(to: UInt8.self), UInt(len))
		})
	do {
		let m = Model.init(failedToLoad: false)
		m.modelCompiledAsset = try MLModelAsset.init(specification: data)
		m.computeUnits = computeUnits
		return m
	} catch {
		let m = Model.init(failedToLoad: true)
		return m
	}
}

func initWithCompiledAssetBatch(
	ptr: UnsafeMutablePointer<UInt8>, len: Int, compute: ComputePlatform
) -> BatchModel {
	var computeUnits: MLComputeUnits
	switch compute {
	case .Cpu:
		computeUnits = .cpuOnly
		break
	case .CpuAndANE:
		computeUnits = .cpuAndNeuralEngine
		break
	case .CpuAndGpu:
		computeUnits = .cpuAndGPU
		break
	}
	let data = Data.init(
		bytesNoCopy: ptr, count: len,
		deallocator: Data.Deallocator.custom { ptr, len in
			rust_vec_free_u8(ptr.assumingMemoryBound(to: UInt8.self), UInt(len))
		})
	do {
		let m = BatchModel.init(failedToLoad: false)
		m.modelCompiledAsset = try MLModelAsset.init(specification: data)
		m.computeUnits = computeUnits
		return m
	} catch {
		let m = BatchModel.init(failedToLoad: true)
		return m
	}
}

func initWithPath(path: RustString, compute: ComputePlatform, compiled: Bool) -> Model {
	var computeUnits: MLComputeUnits
	switch compute {
	case .Cpu:
		computeUnits = .cpuOnly
		break
	case .CpuAndANE:
		computeUnits = .cpuAndNeuralEngine
		break
	case .CpuAndGpu:
		computeUnits = .cpuAndGPU
		break
	}
	var compiledPath: URL
	if compiled {
		compiledPath = URL(fileURLWithPath: path.toString())
	} else {
		let url = URL(fileURLWithPath: path.toString())
		do {
			compiledPath = try MLModel.compileModel(at: url)
		} catch {
			print("[CoreML compile error] \(error)")
			return Model.init(failedToLoad: true)
		}
	}
	let m = Model.init(failedToLoad: false)
	m.compiledPath = compiledPath
	m.computeUnits = computeUnits
	return m
}

// Compile model and overwrite the file to the permanent location, replacing it if necessary
func compileToPath(model: RustString, to: RustString, name: RustString) -> Bool {
	let url = URL(fileURLWithPath: model.toString())
	do {
		let compiledPath = try MLModel.compileModel(at: url)
		let fileManager = FileManager.default
		let appSupportURL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
		print(appSupportURL)
		let permanentURL = appSupportURL.appendingPathComponent(name.toString())
		_ = try fileManager.replaceItemAt(permanentURL, withItemAt: compiledPath)
	} catch {
		return false
	}
	return true
}

func initWithPathBatch(path: RustString, compute: ComputePlatform, compiled: Bool) -> BatchModel {
	var computeUnits: MLComputeUnits
	switch compute {
	case .Cpu:
		computeUnits = .cpuOnly
		break
	case .CpuAndANE:
		computeUnits = .cpuAndNeuralEngine
		break
	case .CpuAndGpu:
		computeUnits = .cpuAndGPU
		break
	}
	var compiledPath: URL
	if compiled {
		compiledPath = URL(fileURLWithPath: path.toString())
	} else {
		let url = URL(fileURLWithPath: path.toString())
		do {
			compiledPath = try MLModel.compileModel(at: url)
		} catch {
			return BatchModel.init(failedToLoad: true)
		}
	}
	let m = BatchModel.init(failedToLoad: false)
	m.compiledPath = compiledPath
	m.computeUnits = computeUnits
	return m
}

struct RuntimeError: LocalizedError {
	let description: String

	init(_ description: String) {
		self.description = description
	}

	var errorDescription: String? {
		description
	}
}

class Model: @unchecked Sendable {
	var compiledPath: URL? = nil
	var modelCompiledAsset: MLModelAsset? = nil
	var model: MLModel? = nil
	var dict: [String: Any] = [:]
	var outputs: [String: Any] = [:]
	var computeUnits: MLComputeUnits = .cpuAndNeuralEngine
	var allowLowPrecisionAccumulationOnGPU: Bool? = nil
	var predictionUsesCPUOnly: Bool? = nil
	var state: Any? = nil  // MLState (macOS 15+), stored as Any for backwards compat

	var failedToLoad: Bool
	init(failedToLoad: Bool) {
		self.failedToLoad = failedToLoad
	}

	func getCompiledPath() -> RustString? {
		return self.compiledPath?.absoluteString.intoRustString()
	}

	func hasFailedToLoad() -> Bool {
		return self.failedToLoad
	}

	func setAllowLowPrecisionAccumulationOnGPU(enabled: Bool) {
		self.allowLowPrecisionAccumulationOnGPU = enabled
	}

	func setPredictionUsesCPUOnly(enabled: Bool) {
		self.predictionUsesCPUOnly = enabled
	}

	func load() -> Bool {
		if hasFailedToLoad() { return false }
		let config = MLModelConfiguration.init()
		config.computeUnits = self.computeUnits
		if let enabled = self.allowLowPrecisionAccumulationOnGPU {
			config.allowLowPrecisionAccumulationOnGPU = enabled
		}
		do {
			if self.compiledPath == nil {
				let semaphore = DispatchSemaphore(value: 0)
				Task { [weak self] in
					guard let self else { return }
					let asset = self.modelCompiledAsset!
					let res = try await MLModel.load(asset: asset, configuration: config)
					self.model = res
					semaphore.signal()
				}
				semaphore.wait()
			} else {
				let loadedModel = try MLModel(contentsOf: self.compiledPath!, configuration: config)
				self.model = loadedModel
			}
			return true
		} catch {
			print("[CoreML load error] \(error)")
			return false
		}
	}

	func unload() -> Bool {
		if hasFailedToLoad() { return false }
		self.model = nil
		self.state = nil
		self.dict = [:]
		self.outputs = [:]
		self.previousDict = [:]
		self.previousOutputs = [:]
		return true
	}

	// MARK: - CoreML State (MLState) for stateful KV cache

	func makeState() -> Bool {
		guard let model = self.model else { return false }
		if #available(macOS 15.0, iOS 18.0, *) {
			self.state = model.makeState()
			return true
		} else {
			return false
		}
	}

	private func predictionOptions() -> MLPredictionOptions {
		let opts = MLPredictionOptions.init()
		if let usesCPUOnly = self.predictionUsesCPUOnly {
			opts.usesCPUOnly = usesCPUOnly
		}
		opts.outputBackings = self.outputs
		return opts
	}

	// Keep previous prediction's buffers alive to prevent ANE use-after-free.
	// CoreML's ANE pipeline may still reference buffers asynchronously.
	var previousDict: [String: Any] = [:]
	var previousOutputs: [String: Any] = [:]

	private func finalizePredictionOutput(from result: MLFeatureProvider) -> ModelOutput {
		let usedOutputBackings = !self.outputs.isEmpty
		let outputs: [String: Any]
		if usedOutputBackings {
			outputs = self.outputs
		} else {
			let features = result as? MLDictionaryFeatureProvider
			outputs = features?.dictionary ?? [:]
		}
		// Keep previous buffers alive — freed on NEXT prediction when replaced.
		self.previousDict = self.dict
		self.previousOutputs = self.outputs
		self.outputs = [:]
		self.dict = [:]
		return ModelOutput(output: outputs, error: nil, cpy: true)
	}

	func predictWithState() -> ModelOutput {
		guard let model = self.model, let stateAny = self.state else {
			return ModelOutput(output: nil, error: RuntimeError("Model or state not loaded"))
		}
		do {
			let input = try MLDictionaryFeatureProvider.init(dictionary: self.dict)
			let opts = predictionOptions()
			if #available(macOS 15.0, iOS 18.0, *) {
				guard let state = stateAny as? MLState else {
					return ModelOutput(output: nil, error: RuntimeError("State is not MLState"))
				}
				let result = try model.prediction(from: input, using: state, options: opts)
				return finalizePredictionOutput(from: result)
			} else {
				return ModelOutput(output: nil, error: RuntimeError("MLState requires macOS 15+"))
			}
		} catch {
			// Clear IOSurface-backed outputs on failure (#862 review follow-up).
			self.previousDict = self.dict
			self.previousOutputs = self.outputs
			self.outputs = [:]
			self.dict = [:]
			return ModelOutput(output: nil, error: error)
		}
	}

	func hasState() -> Bool {
		return self.state != nil
	}

	func resetState() {
		self.state = nil
	}

	func description() -> ModelDescription {
		return ModelDescription(desc: self.model?.modelDescription)
	}

	func bindOutputF32(
		shape: RustVec<Int32>, featureName: RustStr, data: UnsafeMutablePointer<Float32>,
		len: UInt
	) -> Bool {
		if hasFailedToLoad() { return false }
		do {
			var arr: [NSNumber] = []
			var stride: [NSNumber] = []
			var m: Int32 = 1
			for i in shape.reversed() {
				stride.append(NSNumber(value: m))
				m = i * m
			}
			stride.reverse()
			for s in shape {
				arr.append(NSNumber(value: s))
			}
			let deallocMultiArrayRust = { (_ ptr: UnsafeMutableRawPointer) in
				()
			}
			let array = try MLMultiArray.init(
				dataPointer: data, shape: arr, dataType: MLMultiArrayDataType.float32,
				strides: stride, deallocator: deallocMultiArrayRust)
			self.outputs[featureName.toString()] = array
			return true
		} catch {
			print("Unexpected output error: \(error)")
			return false
		}
	}

	func bindOutputU16(
		shape: RustVec<Int32>, featureName: RustStr, data: UnsafeMutablePointer<UInt16>,
		len: UInt
	) -> Bool {
		if hasFailedToLoad() { return false }
		do {
			var arr: [NSNumber] = []
			var stride: [NSNumber] = []
			var m: Int32 = 1
			for i in shape.reversed() {
				stride.append(NSNumber(value: m))
				m = i * m
			}
			stride.reverse()
			for s in shape {
				arr.append(NSNumber(value: s))
			}
			let deallocMultiArrayRust = { (_ ptr: UnsafeMutableRawPointer) in
				()
			}
			let array = try MLMultiArray.init(
				dataPointer: data, shape: arr, dataType: MLMultiArrayDataType.float16,
				strides: stride, deallocator: deallocMultiArrayRust)
			self.outputs[featureName.toString()] = array
			return true
		} catch {
			print("Unexpected output error: \(error)")
			return false
		}
	}

	func bindOutputI32(
		shape: RustVec<Int32>, featureName: RustStr, data: UnsafeMutablePointer<Int32>,
		len: UInt
	) -> Bool {
		if hasFailedToLoad() { return false }
		do {
			var arr: [NSNumber] = []
			var stride: [NSNumber] = []
			var m: Int32 = 1
			for i in shape.reversed() {
				stride.append(NSNumber(value: m))
				m = i * m
			}
			stride.reverse()
			for s in shape {
				arr.append(NSNumber(value: s))
			}
			let deallocMultiArrayRust = { (_ ptr: UnsafeMutableRawPointer) in
				()
			}
			let array = try MLMultiArray.init(
				dataPointer: data, shape: arr, dataType: MLMultiArrayDataType.int32,
				strides: stride, deallocator: deallocMultiArrayRust)
			self.outputs[featureName.toString()] = array
			return true
		} catch {
			print("Unexpected output error: \(error)")
			return false
		}
	}

	func predict() -> ModelOutput {
		if hasFailedToLoad() {
			return ModelOutput(
				output: nil, error: RuntimeError("Failed to load model; can't run predict"))
		}
		do {
			let input = try MLDictionaryFeatureProvider.init(dictionary: self.dict)
			let opts = predictionOptions()

			let result = try self.model!.prediction(from: input, options: opts)
			return finalizePredictionOutput(from: result)
		} catch {
			// Clear IOSurface-backed outputs on failure to prevent leaked bindings.
			// finalizePredictionOutput is only called on success, so the error path
			// must release references explicitly (#862 review follow-up).
			self.previousDict = self.dict
			self.previousOutputs = self.outputs
			self.outputs = [:]
			self.dict = [:]
			return ModelOutput(output: nil, error: error)
		}
	}

	func bindInputF32(
		shape: RustVec<UInt>, featureName: RustStr, data: UnsafeMutablePointer<Float32>,
		len: UInt
	) -> Bool {
		do {
			var arr: [NSNumber] = []
			var stride: [NSNumber] = []
			var m: UInt = 1
			for i in shape.reversed() {
				stride.append(NSNumber(value: m))
				m = i * m
			}
			stride.reverse()
			for s in shape {
				arr.append(NSNumber(value: s))
			}
			let deallocMultiArrayRust = { (_ ptr: UnsafeMutableRawPointer) in
				rust_vec_free_f32(ptr.assumingMemoryBound(to: Float32.self), len)
			}
			let array = try MLMultiArray.init(
				dataPointer: data, shape: arr, dataType: MLMultiArrayDataType.float32,
				strides: stride, deallocator: deallocMultiArrayRust)
			let value = MLFeatureValue(multiArray: array)
			self.dict[featureName.toString()] = value
			return true
		} catch {
			print("Unexpected input error; \(error)")
			return false
		}
	}

	func bindInputI32(
		shape: RustVec<UInt>, featureName: RustStr, data: UnsafeMutablePointer<Int32>, len: UInt
	) -> Bool {
		do {
			var arr: [NSNumber] = []
			var stride: [NSNumber] = []
			var m: UInt = 1
			for i in shape.reversed() {
				stride.append(NSNumber(value: m))
				m = i * m
			}
			stride.reverse()
			for s in shape {
				arr.append(NSNumber(value: s))
			}
			let deallocMultiArrayRust = { (_ ptr: UnsafeMutableRawPointer) -> Void in
				rust_vec_free_i32(ptr.assumingMemoryBound(to: Int32.self), len)
			}
			let array = try MLMultiArray.init(
				dataPointer: data, shape: arr, dataType: MLMultiArrayDataType.int32,
				strides: stride, deallocator: deallocMultiArrayRust)
			let value = MLFeatureValue(multiArray: array)
			self.dict[featureName.toString()] = value
			return true
		} catch {
			print("Unexpected error; \(error)")
			return false
		}
	}

	func bindInputU16(
		shape: RustVec<UInt>, featureName: RustStr, data: UnsafeMutablePointer<UInt16>,
		len: UInt
	) -> Bool {
		do {
			var arr: [NSNumber] = []
			var stride: [NSNumber] = []
			var m: UInt = 1
			for i in shape.reversed() {
				stride.append(NSNumber(value: m))
				m = i * m
			}
			stride.reverse()
			for s in shape {
				arr.append(NSNumber(value: s))
			}
			let deallocMultiArrayRust = { (_ ptr: UnsafeMutableRawPointer) -> Void in
				rust_vec_free_u16(ptr.assumingMemoryBound(to: UInt16.self), len)
			}
			let array = try MLMultiArray.init(
				dataPointer: data, shape: arr, dataType: MLMultiArrayDataType.float16,
				strides: stride, deallocator: deallocMultiArrayRust)
			let value = MLFeatureValue(multiArray: array)
			self.dict[featureName.toString()] = value
			return true
		} catch {
			print("Unexpected error; \(error)")
			return false
		}
	}

	func bindInputCVPixelBuffer(
		width: UInt, height: UInt, featureName: RustStr, data: UnsafeMutablePointer<UInt8>,
		len: UInt
	) -> Bool {
		do {
			// Create CVPixelBuffer from raw BGRA data
			var pixelBuffer: CVPixelBuffer? = nil
			let bytesPerRow = width * 4  // 4 bytes per pixel (BGRA)

			// Create a context to hold the length
			let contextPtr = UnsafeMutablePointer<UInt>.allocate(capacity: 1)
			contextPtr.pointee = len

			// Create pixel buffer with the provided data
			let status = CVPixelBufferCreateWithBytes(
				nil,
				Int(width),
				Int(height),
				kCVPixelFormatType_32BGRA,
				data,
				Int(bytesPerRow),
				{ releaseContext, baseAddress in
					// Deallocator callback - free the Rust vec when CVPixelBuffer is done
					if let baseAddress = baseAddress, let releaseContext = releaseContext {
						let mutablePtr = UnsafeMutablePointer<UInt8>(mutating: baseAddress.assumingMemoryBound(to: UInt8.self))
						let len = releaseContext.assumingMemoryBound(to: UInt.self).pointee
						rust_vec_free_u8(mutablePtr, len)
						// Free the context pointer
						releaseContext.assumingMemoryBound(to: UInt.self).deallocate()
					}
				},
				contextPtr,
				nil,
				&pixelBuffer
			)

			guard status == kCVReturnSuccess, let pixelBuffer = pixelBuffer else {
				print("Failed to create CVPixelBuffer: \(status)")
				contextPtr.deallocate()
				return false
			}

			let value = MLFeatureValue(pixelBuffer: pixelBuffer)
			self.dict[featureName.toString()] = value
			return true
		} catch {
			print("Unexpected CVPixelBuffer error: \(error)")
			return false
		}
	}

	// #828 P0a: zero-copy IOSurface input binding.
	//
	// `surface` is an IOSurfaceRef passed as a raw pointer. The caller
	// is responsible for locking the surface before calling this
	// function and keeping it locked through the subsequent predict()
	// call. We retain the surface for the MLMultiArray lifetime; the
	// deallocator closure releases it when CoreML is done with the
	// array.
	//
	// `dtypeRaw` matches `MLDataType::raw_tag()` on the Rust side:
	//   0 -> Float32, 1 -> Float16, 2 -> Int32.
	func bindInputIOSurface(
		surface: UnsafeMutablePointer<UInt8>,
		dtypeRaw: Int32,
		shape: RustVec<UInt>,
		featureName: RustStr
	) -> Bool {
		do {
			// Reinterpret the opaque pointer as an IOSurfaceRef.
			let rawPtr = UnsafeRawPointer(surface)
			let surfaceRef = Unmanaged<IOSurface>.fromOpaque(rawPtr).takeUnretainedValue()

			// Select MLMultiArrayDataType from the Rust-side tag.
			let dataType: MLMultiArrayDataType
			switch dtypeRaw {
			case 0:
				dataType = .float32
			case 1:
				dataType = .float16
			case 2:
				dataType = .int32
			default:
				print("bindInputIOSurface: unknown dtype tag \(dtypeRaw)")
				return false
			}

			// Build shape + contiguous row-major strides.
			var shapeArr: [NSNumber] = []
			var strideArr: [NSNumber] = []
			var m: UInt = 1
			for i in shape.reversed() {
				strideArr.append(NSNumber(value: m))
				m = i * m
			}
			strideArr.reverse()
			for s in shape {
				shapeArr.append(NSNumber(value: s))
			}

			// Base address must be read while the caller holds the lock.
			let baseAddress = IOSurfaceGetBaseAddress(surfaceRef)

			// Retain the IOSurface for the MLMultiArray's lifetime. The
			// deallocator releases it; the pointer itself belongs to the
			// IOSurface, not to the heap, so we must NOT free() it.
			let retained = Unmanaged.passRetained(surfaceRef)
			let deallocMultiArray = { (_ ptr: UnsafeMutableRawPointer) -> Void in
				retained.release()
			}

			let array = try MLMultiArray(
				dataPointer: baseAddress,
				shape: shapeArr,
				dataType: dataType,
				strides: strideArr,
				deallocator: deallocMultiArray
			)
			let value = MLFeatureValue(multiArray: array)
			self.dict[featureName.toString()] = value
			return true
		} catch {
			print("Unexpected IOSurface input error: \(error)")
			return false
		}
	}

	// #828 P0a: zero-copy CVPixelBuffer input binding (borrow path).
	//
	// Unlike `bindInputCVPixelBuffer`, this does not take ownership of
	// a Rust-side Vec<u8>. It wraps the passed CVPixelBufferRef in an
	// MLFeatureValue(pixelBuffer:) directly — Swift retains the pixel
	// buffer for the lifetime of the feature value.
	func bindInputCVPixelBufferRef(
		pixelBuffer: UnsafeMutablePointer<UInt8>,
		featureName: RustStr
	) -> Bool {
		let rawPtr = UnsafeRawPointer(pixelBuffer)
		let pixelBufferRef = Unmanaged<CVPixelBuffer>.fromOpaque(rawPtr).takeUnretainedValue()
		let value = MLFeatureValue(pixelBuffer: pixelBufferRef)
		self.dict[featureName.toString()] = value
		return true
	}

	// #828 P0d: zero-copy IOSurface OUTPUT binding.
	//
	// Binds a caller-provided IOSurfaceRef as the destination backing
	// for a named model output. CoreML will write the prediction
	// result for `featureName` directly into the surface's base
	// address rather than allocating a fresh MLMultiArray. This is the
	// write-side mirror of `bindInputIOSurface` used by the #828 P1
	// hybrid decode rewrite to chain CoreML FFN output into the next
	// layer's Metal attention input with zero copies.
	//
	// The caller must hold a read-write lock on the surface (NOT
	// `kIOSurfaceLockReadOnly` — CoreML writes) for the duration of
	// the subsequent predict()/predictWithState() call. We retain the
	// surface for the MLMultiArray lifetime; the deallocator releases
	// it when CoreML is done with the array, which happens when
	// `self.outputs` is cleared in `finalizePredictionOutput`.
	//
	// `dtypeRaw` matches `MLDataType::raw_tag()` on the Rust side:
	//   0 -> Float32, 1 -> Float16, 2 -> Int32.
	//
	// The backing installed here flows through the existing
	// `predictionOptions().outputBackings = self.outputs` plumbing —
	// no new MLPredictionOptions work is required.
	func bindOutputIOSurface(
		surface: UnsafeMutablePointer<UInt8>,
		dtypeRaw: Int32,
		shape: RustVec<Int32>,
		featureName: RustStr
	) -> Bool {
		if hasFailedToLoad() { return false }
		do {
			// Reinterpret the opaque pointer as an IOSurfaceRef.
			let rawPtr = UnsafeRawPointer(surface)
			let surfaceRef = Unmanaged<IOSurface>.fromOpaque(rawPtr).takeUnretainedValue()

			// Select MLMultiArrayDataType from the Rust-side tag.
			let dataType: MLMultiArrayDataType
			switch dtypeRaw {
			case 0:
				dataType = .float32
			case 1:
				dataType = .float16
			case 2:
				dataType = .int32
			default:
				print("bindOutputIOSurface: unknown dtype tag \(dtypeRaw)")
				return false
			}

			// Build shape + contiguous row-major strides. Shape uses
			// Int32 here to match the other bindOutput* functions
			// (bindOutputF32/U16/I32).
			var shapeArr: [NSNumber] = []
			var strideArr: [NSNumber] = []
			var m: Int32 = 1
			for i in shape.reversed() {
				strideArr.append(NSNumber(value: m))
				m = i * m
			}
			strideArr.reverse()
			for s in shape {
				shapeArr.append(NSNumber(value: s))
			}

			// Base address must be read while the caller holds the
			// read-write lock. CoreML will write into this pointer
			// during predict().
			let baseAddress = IOSurfaceGetBaseAddress(surfaceRef)

			// Retain the IOSurface for the MLMultiArray's lifetime.
			// The deallocator releases it; the pointer itself belongs
			// to the IOSurface, not the heap, so we must NOT free() it.
			let retained = Unmanaged.passRetained(surfaceRef)
			let deallocMultiArray = { (_ ptr: UnsafeMutableRawPointer) -> Void in
				retained.release()
			}

			let array = try MLMultiArray(
				dataPointer: baseAddress,
				shape: shapeArr,
				dataType: dataType,
				strides: strideArr,
				deallocator: deallocMultiArray
			)
			// Install as an output backing. `predictionOptions()` copies
			// `self.outputs` into `MLPredictionOptions.outputBackings`
			// before predict().
			self.outputs[featureName.toString()] = array
			return true
		} catch {
			print("Unexpected IOSurface output error: \(error)")
			return false
		}
	}
}
