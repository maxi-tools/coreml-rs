import CoreML

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
	var failedToLoad: Bool

	init(failedToLoad: Bool = false, model: MLModel? = nil) {
		self.failedToLoad = failedToLoad
	}

	func hasFailedToLoad() -> Bool {
		return self.failedToLoad
	}

	func description() -> ModelDescription {
		return ModelDescription(desc: self.model?.modelDescription)
	}

	func load() -> Bool {
		if hasFailedToLoad() { return false }
		let config = MLModelConfiguration.init()
		config.computeUnits = self.computeUnits
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
		shape: RustVec<UInt>, featureName: RustString, data: UnsafeMutablePointer<Float32>,
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
	func output_type(name: RustString) -> RustString {
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
	func output_shape(name: RustString) -> RustVec<UInt> {
		if !failedToLoad() {
			let res = self.description?.outputDescriptionsByName[name.toString()]
			guard let res else { return RustVec.init() }
			let arr = res.multiArrayConstraint
			guard let arr else { return RustVec.init() }
			let ret = RustVec<UInt>()
			// Check for flexible dimensions via shapeConstraint range (same as input_shape)
			let ranges = arr.shapeConstraint.sizeRangeForDimension
			for (i, r) in arr.shape.enumerated() {
				if i < ranges.count {
					let nsRange = ranges[i].rangeValue
					if nsRange.length > 0 {
						ret.push(value: 0)
						continue
					}
				}
				ret.push(value: UInt(truncating: r))
			}
			return ret
		}
		return RustVec.init()
	}
	func input_shape(name: RustString) -> RustVec<UInt> {
		if !failedToLoad() {
			let res = self.description?.inputDescriptionsByName[name.toString()]
			guard let res else { return RustVec.init() }
			let arr = res.multiArrayConstraint
			guard let arr else { return RustVec.init() }
			let ret = RustVec<UInt>()
			// Check for flexible dimensions via shapeConstraint range
			let ranges = arr.shapeConstraint.sizeRangeForDimension
			for (i, r) in arr.shape.enumerated() {
				if i < ranges.count {
					let nsRange = ranges[i].rangeValue  // NSRange: location=min, length=max-min
					if nsRange.length > 0 {
						// Flexible dim — return 0 as wildcard
						ret.push(value: 0)
						continue
					}
				}
				ret.push(value: UInt(truncating: r))
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
}

// Check if MLMultiArray strides match contiguous (C-order) layout
func isContiguous(shape: [Int], strides: [Int]) -> Bool {
	guard shape.count == strides.count else { return false }
	var expected = 1
	for i in (0..<shape.count).reversed() {
		if strides[i] != expected { return false }
		expected *= shape[i]
	}
	return true
}

// Copy from strided source to contiguous destination buffer (pointer arithmetic, no NSNumber)
func copyDestrided<T>(src: UnsafePointer<T>, dst: UnsafeMutablePointer<T>, shape: [Int], strides: [Int], name: String = "") {
	if !name.isEmpty {
		var expected: [Int] = Array(repeating: 1, count: shape.count)
		for i in (0..<shape.count - 1).reversed() {
			expected[i] = expected[i + 1] * shape[i + 1]
		}
		let count = shape.reduce(1, *)
		print("[coreml-rs] de-striding \(name): shape=\(shape) strides=\(strides) expected=\(expected) count=\(count)")
	}
	let ndim = shape.count
	if ndim == 0 { return }
	func copyRecursive(srcOffset: Int, dstOffset: inout Int, dim: Int) {
		if dim == ndim - 1 {
			for i in 0..<shape[dim] {
				dst[dstOffset] = src[srcOffset + i * strides[dim]]
				dstOffset += 1
			}
		} else {
			for i in 0..<shape[dim] {
				copyRecursive(srcOffset: srcOffset + i * strides[dim], dstOffset: &dstOffset, dim: dim + 1)
			}
		}
	}
	var dstOff = 0
	copyRecursive(srcOffset: 0, dstOffset: &dstOff, dim: 0)
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
		guard let output = self.output else { return RustVec.init() }
		let ret = RustVec<RustString>()
		for key in output.keys {
			guard let val = output[key] else { continue }
			let str = "\(key):\(val)".intoRustString()
			ret.push(value: str)
		}
		return ret
	}
	func outputShape(name: RustString) -> RustVec<UInt> {
		if hasFailedToLoad() { return RustVec.init() }
		guard let output = self.output, let value = output[name.toString()] else { return RustVec.init() }
		let out: MLMultiArray
		if let feature = value as? MLFeatureValue {
			guard let arr = feature.multiArrayValue else { return RustVec.init() }
			out = arr
		} else if let arr = value as? MLMultiArray {
			out = arr
		} else {
			return RustVec.init()
		}
		let ret = RustVec<UInt>()
		for dim in out.shape {
			ret.push(value: UInt(truncating: dim))
		}
		return ret
	}
		func outputF32(name: RustString) -> RustVec<Float32> {
if hasFailedToLoad() { return RustVec.init() }
guard let output = self.output else { return RustVec.init() }
guard let value = output[name.toString()] else { return RustVec.init() }

let out: MLMultiArray
if let feature = value as? MLFeatureValue {
guard let arr = feature.multiArrayValue else { return RustVec.init() }
out = arr
} else if let arr = value as? MLMultiArray {
out = arr
} else {
return RustVec.init()
}

		let l = out.count
		if l == 0 { return RustVec<Float32>() }
		let shape = out.shape.map { $0.intValue }
		var v = RustVec<Float32>()
		out.withUnsafeMutableBytes { ptr, strides in
			guard let base = ptr.baseAddress else { return }
			let p = base.assumingMemoryBound(to: Float32.self)
			if isContiguous(shape: shape, strides: strides) {
				if self.cpy {
					v = rust_vec_from_ptr_f32_cpy(p, UInt(l))
				} else {
					v = rust_vec_from_ptr_f32(p, UInt(l))
				}
			} else {
				let buf = UnsafeMutablePointer<Float32>.allocate(capacity: l)
				copyDestrided(src: p, dst: buf, shape: shape, strides: strides, name: "outputF32 \(name)")
				v = rust_vec_from_ptr_f32_cpy(buf, UInt(l))
				buf.deallocate()
			}
		}
		return v
	}
		func outputI32(name: RustString) -> RustVec<Int32> {
if hasFailedToLoad() { return RustVec.init() }
guard let output = self.output else { return RustVec.init() }
guard let value = output[name.toString()] else { return RustVec.init() }

let out: MLMultiArray
if let feature = value as? MLFeatureValue {
guard let arr = feature.multiArrayValue else { return RustVec.init() }
out = arr
} else if let arr = value as? MLMultiArray {
out = arr
} else {
return RustVec.init()
}

		let l = out.count
		if l == 0 { return RustVec<Int32>() }
		let shape = out.shape.map { $0.intValue }
		var v = RustVec<Int32>()
		out.withUnsafeMutableBytes { ptr, strides in
			guard let base = ptr.baseAddress else { return }
			let p = base.assumingMemoryBound(to: Int32.self)
			if isContiguous(shape: shape, strides: strides) {
				if self.cpy {
					v = rust_vec_from_ptr_i32_cpy(p, UInt(l))
				} else {
					v = rust_vec_from_ptr_i32(p, UInt(l))
				}
			} else {
				let buf = UnsafeMutablePointer<Int32>.allocate(capacity: l)
				copyDestrided(src: p, dst: buf, shape: shape, strides: strides, name: "outputI32 \(name)")
				v = rust_vec_from_ptr_i32_cpy(buf, UInt(l))
				buf.deallocate()
			}
		}
		return v
	}
		func outputU16(name: RustString) -> RustVec<UInt16> {
		if hasFailedToLoad() { return RustVec.init() }
		guard let output = self.output else { return RustVec.init() }
		guard let value = output[name.toString()] else { return RustVec.init() }

		let out: MLMultiArray
		if let feature = value as? MLFeatureValue {
			guard let arr = feature.multiArrayValue else { return RustVec.init() }
			out = arr
		} else if let arr = value as? MLMultiArray {
			out = arr
		} else {
			return RustVec.init()
		}

		let l = out.count
		if l == 0 { return RustVec<UInt16>() }
		let shape = out.shape.map { $0.intValue }
		var v = RustVec<UInt16>()
		out.withUnsafeMutableBytes { ptr, strides in
			guard let base = ptr.baseAddress else { return }
			let p = base.assumingMemoryBound(to: UInt16.self)
			if isContiguous(shape: shape, strides: strides) {
				if self.cpy {
					v = rust_vec_from_ptr_u16_cpy(p, UInt(l))
				} else {
					v = rust_vec_from_ptr_u16(p, UInt(l))
				}
			} else {
				let buf = UnsafeMutablePointer<UInt16>.allocate(capacity: l)
				copyDestrided(src: p, dst: buf, shape: shape, strides: strides, name: "outputU16 \(name)")
				v = rust_vec_from_ptr_u16_cpy(buf, UInt(l))
				buf.deallocate()
			}
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

	func load() -> Bool {
		if hasFailedToLoad() { return false }
		let config = MLModelConfiguration.init()
		config.computeUnits = self.computeUnits
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

	func predictWithState() -> ModelOutput {
		guard let model = self.model, let stateAny = self.state else {
			return ModelOutput(output: nil, error: RuntimeError("Model or state not loaded"))
		}
		do {
			let input = try MLDictionaryFeatureProvider.init(dictionary: self.dict)
			let opts = MLPredictionOptions.init()
			opts.outputBackings = self.outputs
			if #available(macOS 15.0, iOS 18.0, *) {
				guard let state = stateAny as? MLState else {
					return ModelOutput(output: nil, error: RuntimeError("State is not MLState"))
				}
				let result = try model.prediction(from: input, using: state, options: opts)
				let outputs: [String: Any]
				if self.outputs.isEmpty {
					let features = result as? MLDictionaryFeatureProvider
					outputs = features?.dictionary ?? [:]
				} else {
					outputs = self.outputs
				}
				self.outputs = [:]
				self.dict = [:]
				return ModelOutput(output: outputs, error: nil, cpy: true)
			} else {
				return ModelOutput(output: nil, error: RuntimeError("MLState requires macOS 15+"))
			}
		} catch {
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
		shape: RustVec<Int32>, featureName: RustString, data: UnsafeMutablePointer<Float32>,
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
		shape: RustVec<Int32>, featureName: RustString, data: UnsafeMutablePointer<UInt16>,
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
		shape: RustVec<Int32>, featureName: RustString, data: UnsafeMutablePointer<Int32>,
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
			let opts = MLPredictionOptions.init()
			opts.outputBackings = self.outputs

			let result = try self.model!.prediction(from: input, options: opts)
			
			// If we have output backings, use them
			let outputs: [String: Any]
			if self.outputs.isEmpty {
				// No output backing - extract from prediction result
				let features = result as? MLDictionaryFeatureProvider
				let rawDict = features?.dictionary ?? [:]
				outputs = rawDict
			} else {
				// Use output backing
				outputs = self.outputs
			}
			
			self.outputs = [:]
			self.dict = [:]
			// Use cpy=true to safely copy data from MLFeatureValue outputs
			return ModelOutput(output: outputs, error: nil, cpy: true)
		} catch {
			// print("Unexpected predict error: \(error)")
			return ModelOutput(output: nil, error: error)
		}
	}

	func bindInputF32(
		shape: RustVec<UInt>, featureName: RustString, data: UnsafeMutablePointer<Float32>,
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
		shape: RustVec<UInt>, featureName: RustString, data: UnsafeMutablePointer<Int32>, len: UInt
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
		shape: RustVec<UInt>, featureName: RustString, data: UnsafeMutablePointer<UInt16>,
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
		width: UInt, height: UInt, featureName: RustString, data: UnsafeMutablePointer<UInt8>,
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
}
