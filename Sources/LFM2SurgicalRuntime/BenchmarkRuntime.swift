import CoreML
import CryptoKit
import Foundation

/// One frozen class-contiguous segment and its explicit conv-state inputs.
public struct SegmentDescriptor: Sendable {
    /// Artifact stem under `stage1/segments`.
    public let name: String

    /// Conv-state input names, empty for GQA segments.
    public let stateInputs: [String]

    /// True when the segment consumes the shared RoPE and causal-mask inputs.
    public let usesAttentionInputs: Bool

    /// Creates one immutable descriptor.
    public init(name: String, stateInputs: [String], usesAttentionInputs: Bool) {
        self.name = name
        self.stateInputs = stateInputs
        self.usesAttentionInputs = usesAttentionInputs
    }
}

/// Frozen Hugging Face revision and graph-defining file digests.
public struct CheckpointIdentity: Codable, Sendable {
    /// Upstream Hugging Face repository.
    public let repository: String

    /// Immutable Hugging Face commit.
    public let revision: String

    /// SHA-256 of `model.safetensors`.
    public let modelSha256: String

    /// SHA-256 of `config.json`.
    public let configSha256: String

    /// SHA-256 of `tokenizer.json`.
    public let tokenizerSha256: String

    private enum CodingKeys: String, CodingKey {
        case repository
        case revision
        case modelSha256 = "model_sha256"
        case configSha256 = "config_sha256"
        case tokenizerSha256 = "tokenizer_sha256"
    }
}

/// Exact Stage 1 G1a input written by `prepare_stage1_input.py`.
public struct Stage1Input: Decodable, Sendable {
    /// The frozen prompt bucket; Stage 1 requires 512.
    public let bucket: Int

    /// Exact real-checkpoint identity verified by the Python preparation gate.
    public let checkpoint: CheckpointIdentity

    /// coremltools version that generated the packages and input evidence.
    public let coremltools: String

    /// The first registered real prompt.
    public let prompt: String

    /// Token IDs repeated deterministically to fill the bucket.
    public let tokenIds: [Int32]

    private enum CodingKeys: String, CodingKey {
        case bucket
        case checkpoint
        case coremltools
        case prompt
        case tokenIds = "token_ids"
    }
}

/// One exact package-tree digest included in the authoritative Swift result.
public struct ArtifactIdentity: Codable, Sendable {
    /// Stable experiment label.
    public let name: String

    /// Package path relative to the Stage 1 output directory.
    public let path: String

    /// SHA-256 over sorted relative paths and file bytes.
    public let sha256: String
}

/// Host and toolchain provenance for the Swift timing process.
public struct RuntimeEnvironment: Codable, Sendable {
    /// Hostname reported by the timed process.
    public let host: String

    /// Full operating-system version string.
    public let operatingSystem: String

    /// coremltools version that generated every package.
    public let coremltools: String
}

/// Raw and summarized Swift G1a timing evidence.
public struct G1AResult: Codable, Sendable {
    /// Runtime name proving that Python/NumPy was absent from timed boundaries.
    public let runtime: String

    /// Exact checkpoint identity carried from the fail-closed input fixture.
    public let checkpoint: CheckpointIdentity

    /// Host and conversion-tool provenance.
    public let environment: RuntimeEnvironment

    /// Exact package hashes for every timed Core ML artifact.
    public let artifacts: [ArtifactIdentity]

    /// Frozen prompt bucket.
    public let bucket: Int

    /// Exact registered real prompt repeated to fill the bucket.
    public let prompt: String

    /// Core ML compute-unit permission used by every model.
    public let computeUnits: String

    /// Number of untimed warmups per candidate.
    public let warmupsPerCandidate: Int

    /// Number of counterbalanced measured runs per candidate.
    public let measuredRunsPerCandidate: Int

    /// Alternating candidate order for every pair.
    public let runOrder: [String]

    /// Monolithic end-to-end prefill timings.
    public let monolithicTimesMs: [Double]

    /// Segmented end-to-end prefill timings.
    public let segmentedTimesMs: [Double]

    /// Monolithic median in milliseconds.
    public let monolithicMedianMs: Double

    /// Monolithic inclusive interquartile range in milliseconds.
    public let monolithicIQRMs: Double

    /// Segmented median in milliseconds.
    public let segmentedMedianMs: Double

    /// Segmented inclusive interquartile range in milliseconds.
    public let segmentedIQRMs: Double

    /// Segmented minus monolithic median latency.
    public let overheadMs: Double

    /// Registered G1a fraction: overhead divided by segmented total prefill.
    public let overheadFractionOfSegmented: Double

    /// Secondary expression: overhead divided by monolithic latency.
    public let overheadRelativeToMonolith: Double

    /// Maximum paired final-logit difference, computed outside timed regions.
    public let outputMaxAbs: Double

    /// Actual hidden-output storage type handed directly between Core ML models.
    public let hiddenBoundaryDataType: String

    /// Frozen G1a stop line.
    public let g1aMaximumOverheadFraction: Double

    /// True only when the registered fraction does not exceed 30%.
    public let passed: Bool

    /// `GO` or `KILL` under the frozen gate.
    public let verdict: String
}

/// Shared macOS/iOS Core ML orchestrator for the frozen 13-segment pipeline.
public final class SurgicalPrefillRuntime {
    /// Frozen checkpoint hidden width.
    private static let hiddenSize = 1_024

    /// Frozen attention head width.
    private static let headDimension = 64

    /// Two live causal-convolution values for kernel width three.
    private static let convolutionStateWidth = 2

    /// Frozen checkpoint rotary base.
    private static let ropeTheta = 1_000_000.0

    /// Largest finite negative fp16 attention-mask value.
    private static let causalMaskNegative = -65_504.0

    /// Pre-registered maximum fraction of segmented prefill spent on boundaries.
    private static let g1aMaximumOverheadFraction = 0.30

    /// Maximal same-class runs in exact checkpoint execution order.
    private static let segments = [
        SegmentDescriptor(
            name: "segment_00_conv_0_1",
            stateInputs: ["conv_state_0_in", "conv_state_1_in"],
            usesAttentionInputs: false
        ),
        SegmentDescriptor(name: "segment_01_gqa_2", stateInputs: [], usesAttentionInputs: true),
        SegmentDescriptor(
            name: "segment_02_conv_3_4",
            stateInputs: ["conv_state_3_in", "conv_state_4_in"],
            usesAttentionInputs: false
        ),
        SegmentDescriptor(name: "segment_03_gqa_5", stateInputs: [], usesAttentionInputs: true),
        SegmentDescriptor(
            name: "segment_04_conv_6_7",
            stateInputs: ["conv_state_6_in", "conv_state_7_in"],
            usesAttentionInputs: false
        ),
        SegmentDescriptor(name: "segment_05_gqa_8", stateInputs: [], usesAttentionInputs: true),
        SegmentDescriptor(name: "segment_06_conv_9", stateInputs: ["conv_state_9_in"], usesAttentionInputs: false),
        SegmentDescriptor(name: "segment_07_gqa_10", stateInputs: [], usesAttentionInputs: true),
        SegmentDescriptor(name: "segment_08_conv_11", stateInputs: ["conv_state_11_in"], usesAttentionInputs: false),
        SegmentDescriptor(name: "segment_09_gqa_12", stateInputs: [], usesAttentionInputs: true),
        SegmentDescriptor(name: "segment_10_conv_13", stateInputs: ["conv_state_13_in"], usesAttentionInputs: false),
        SegmentDescriptor(name: "segment_11_gqa_14", stateInputs: [], usesAttentionInputs: true),
        SegmentDescriptor(name: "segment_12_conv_15", stateInputs: ["conv_state_15_in"], usesAttentionInputs: false),
    ]

    /// Exact prompt and token IDs used by both candidates.
    private let input: Stage1Input

    /// Root used to resolve and hash every generated Core ML package.
    private let stage1Directory: URL

    /// Common token embedding model.
    private let embedding: MLModel

    /// Common final normalization and tied projection model.
    private let languageModelHead: MLModel

    /// One-prediction fixed-512 layer-stack control.
    private let monolithicLayerStack: MLModel

    /// Thirteen class-contiguous layer-stack models in execution order.
    private let segmentedLayerStack: [(SegmentDescriptor, MLModel)]

    /// Reusable int32 token-ID tensor.
    private let tokenIDs: MLMultiArray

    /// Reusable fp16 token mask consumed by the embedding.
    private let tokenMask: MLMultiArray

    /// Reusable fp32 rotary cosine table.
    private let cosine: MLMultiArray

    /// Reusable fp32 rotary sine table.
    private let sine: MLMultiArray

    /// Reusable fp16 finite causal mask.
    private let attentionMask: MLMultiArray

    /// Reusable zeroed explicit state for every prefill conv layer.
    private let zeroState: MLMultiArray

    /// Compiles and loads every package before any timed prediction.
    public init(stage1Directory: URL, inputURL: URL) throws {
        input = try JSONDecoder().decode(Stage1Input.self, from: Data(contentsOf: inputURL))
        self.stage1Directory = stage1Directory
        guard input.bucket == 512, input.tokenIds.count == input.bucket else {
            throw RuntimeError.invalidInput(
                "G1a requires exactly 512 token IDs; found \(input.tokenIds.count)"
            )
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU
        embedding = try Self.loadModel(
            package: stage1Directory.appendingPathComponent("embedding.mlpackage"),
            configuration: configuration
        )
        languageModelHead = try Self.loadModel(
            package: stage1Directory.appendingPathComponent("lm_head.mlpackage"),
            configuration: configuration
        )
        monolithicLayerStack = try Self.loadModel(
            package: stage1Directory.appendingPathComponent(
                "monolithic_layer_stack_512.mlpackage"
            ),
            configuration: configuration
        )
        segmentedLayerStack = try Self.segments.map { descriptor in
            let package = stage1Directory
                .appendingPathComponent("segments", isDirectory: true)
                .appendingPathComponent("\(descriptor.name).mlpackage")
            return (descriptor, try Self.loadModel(package: package, configuration: configuration))
        }
        tokenIDs = try Self.makeTokenIDs(input.tokenIds)
        tokenMask = try Self.makeFilledArray(
            shape: [1, 1, 1, NSNumber(value: input.bucket)],
            dataType: .float16,
            value: 1.0
        )
        (cosine, sine) = try Self.makeRoPETables(sequenceLength: input.bucket)
        attentionMask = try Self.makeCausalMask(sequenceLength: input.bucket)
        zeroState = try Self.makeFilledArray(
            shape: [1, Self.hiddenSize, Self.convolutionStateWidth].map(NSNumber.init),
            dataType: .float16,
            value: 0.0
        )
    }

    /// Runs a counterbalanced, warmed G1a comparison entirely through Swift.
    public func benchmark(warmups: Int, runs: Int) throws -> G1AResult {
        guard warmups > 0, runs >= 4 else {
            throw RuntimeError.invalidInput("warmups must be positive and runs at least four")
        }
        for _ in 0..<warmups {
            _ = try runMonolithic()
            _ = try runSegmented()
        }

        var monolithicTimes: [Double] = []
        var segmentedTimes: [Double] = []
        var runOrder: [String] = []
        var outputMaximumAbsoluteDifference = 0.0
        var hiddenBoundaryDataType = "unknown"
        for runIndex in 0..<runs {
            let order = runIndex.isMultiple(of: 2)
                ? ["monolithic", "segmented"]
                : ["segmented", "monolithic"]
            runOrder.append(order.joined(separator: "->"))
            var outputs: [String: MLMultiArray] = [:]
            for candidate in order {
                let start = DispatchTime.now().uptimeNanoseconds
                let result = candidate == "monolithic"
                    ? try runMonolithic()
                    : try runSegmented()
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000.0
                outputs[candidate] = result.logits
                hiddenBoundaryDataType = result.hiddenBoundaryDataType
                if candidate == "monolithic" {
                    monolithicTimes.append(elapsed)
                } else {
                    segmentedTimes.append(elapsed)
                }
            }
            outputMaximumAbsoluteDifference = max(
                outputMaximumAbsoluteDifference,
                try Self.maximumAbsoluteDifference(
                    outputs["monolithic"]!, outputs["segmented"]!
                )
            )
        }

        let monolithicSummary = Self.summary(monolithicTimes)
        let segmentedSummary = Self.summary(segmentedTimes)
        let overhead = segmentedSummary.median - monolithicSummary.median
        let fraction = overhead / segmentedSummary.median
        let relative = overhead / monolithicSummary.median
        let passed = fraction <= Self.g1aMaximumOverheadFraction
        return G1AResult(
            runtime: "Swift CoreML direct MLMultiArray handoff",
            checkpoint: input.checkpoint,
            environment: RuntimeEnvironment(
                host: ProcessInfo.processInfo.hostName,
                operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
                coremltools: input.coremltools
            ),
            artifacts: try artifactIdentities(),
            bucket: input.bucket,
            prompt: input.prompt,
            computeUnits: "CPU_AND_GPU",
            warmupsPerCandidate: warmups,
            measuredRunsPerCandidate: runs,
            runOrder: runOrder,
            monolithicTimesMs: monolithicTimes,
            segmentedTimesMs: segmentedTimes,
            monolithicMedianMs: monolithicSummary.median,
            monolithicIQRMs: monolithicSummary.iqr,
            segmentedMedianMs: segmentedSummary.median,
            segmentedIQRMs: segmentedSummary.iqr,
            overheadMs: overhead,
            overheadFractionOfSegmented: fraction,
            overheadRelativeToMonolith: relative,
            outputMaxAbs: outputMaximumAbsoluteDifference,
            hiddenBoundaryDataType: hiddenBoundaryDataType,
            g1aMaximumOverheadFraction: Self.g1aMaximumOverheadFraction,
            passed: passed,
            verdict: passed ? "GO" : "KILL"
        )
    }

    /// Runs the common embedding, one layer-stack prediction, and common head.
    private func runMonolithic() throws -> (logits: MLMultiArray, hiddenBoundaryDataType: String) {
        let hidden = try predict(
            model: embedding,
            values: ["token_ids": tokenIDs, "token_mask": tokenMask],
            output: "hidden_out"
        )
        var values = attentionValues(hidden: hidden)
        for descriptor in Self.segments where !descriptor.usesAttentionInputs {
            for stateInput in descriptor.stateInputs {
                values[stateInput] = zeroState
            }
        }
        let output = try predict(
            model: monolithicLayerStack,
            values: values,
            output: "hidden_out"
        )
        try Self.requireFloat16Boundary(output, label: "monolithic hidden output")
        let logits = try runHead(hidden: output)
        return (logits, Self.dataTypeName(output.dataType))
    }

    /// Runs the common embedding, thirteen direct handoffs, and common head.
    private func runSegmented() throws -> (logits: MLMultiArray, hiddenBoundaryDataType: String) {
        var hidden = try predict(
            model: embedding,
            values: ["token_ids": tokenIDs, "token_mask": tokenMask],
            output: "hidden_out"
        )
        try Self.requireFloat16Boundary(hidden, label: "embedding hidden output")
        for (descriptor, model) in segmentedLayerStack {
            var values: [String: MLMultiArray] = ["hidden_states": hidden]
            if descriptor.usesAttentionInputs {
                values.merge(attentionValues(hidden: hidden)) { _, replacement in replacement }
            } else {
                for stateInput in descriptor.stateInputs {
                    values[stateInput] = zeroState
                }
            }
            hidden = try predict(model: model, values: values, output: "hidden_out")
            try Self.requireFloat16Boundary(
                hidden, label: "\(descriptor.name) hidden output"
            )
        }
        return (try runHead(hidden: hidden), Self.dataTypeName(hidden.dataType))
    }

    /// Hashes every timed package and returns a stable provenance list.
    private func artifactIdentities() throws -> [ArtifactIdentity] {
        let relativePaths = [
            "embedding.mlpackage",
            "lm_head.mlpackage",
            "monolithic_layer_stack_512.mlpackage",
        ] + Self.segments.map { "segments/\($0.name).mlpackage" }
        return try relativePaths.map { relativePath in
            let package = stage1Directory.appendingPathComponent(relativePath)
            return ArtifactIdentity(
                name: package.deletingPathExtension().lastPathComponent,
                path: relativePath,
                sha256: try Self.packageTreeSHA256(package)
            )
        }
    }

    /// Keeps shared attention inputs identical for the monolith and segments.
    private func attentionValues(hidden: MLMultiArray) -> [String: MLMultiArray] {
        [
            "hidden_states": hidden,
            "cosine": cosine,
            "sine": sine,
            "attention_mask": attentionMask,
        ]
    }

    /// Copies the final token once and runs the common tied language-model head.
    private func runHead(hidden: MLMultiArray) throws -> MLMultiArray {
        let lastHidden = try MLMultiArray(
            shape: [1, Self.hiddenSize, 1, 1].map(NSNumber.init),
            dataType: .float16
        )
        let lastPosition = input.bucket - 1
        for channel in 0..<Self.hiddenSize {
            lastHidden[[0, channel, 0, 0].map(NSNumber.init)] =
                hidden[[0, channel, 0, lastPosition].map(NSNumber.init)]
        }
        return try predict(
            model: languageModelHead,
            values: ["hidden_states": lastHidden],
            output: "logits"
        )
    }

    /// Runs one Core ML model and fails closed when its documented output is absent.
    private func predict(
        model: MLModel,
        values: [String: MLMultiArray],
        output outputName: String
    ) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: values)
        let prediction = try model.prediction(from: provider)
        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue else {
            throw RuntimeError.missingOutput(outputName)
        }
        return output
    }

    /// Compiles package metadata once, outside every timed region.
    private static func loadModel(
        package: URL, configuration: MLModelConfiguration
    ) throws -> MLModel {
        let compiled = try MLModel.compileModel(at: package)
        return try MLModel(contentsOf: compiled, configuration: configuration)
    }

    /// Creates the exact int32 embedding input from the frozen JSON fixture.
    private static func makeTokenIDs(_ values: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: values.count)], dataType: .int32
        )
        for (index, value) in values.enumerated() {
            array[[0, index].map(NSNumber.init)] = NSNumber(value: value)
        }
        return array
    }

    /// Creates a small reusable constant input outside timed prediction.
    private static func makeFilledArray(
        shape: [NSNumber], dataType: MLMultiArrayDataType, value: Double
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: dataType)
        for index in 0..<array.count {
            array[index] = NSNumber(value: value)
        }
        return array
    }

    /// Reproduces the Python full-sequence RoPE tables in fp32.
    private static func makeRoPETables(
        sequenceLength: Int
    ) throws -> (MLMultiArray, MLMultiArray) {
        let shape = [1, sequenceLength, headDimension].map(NSNumber.init)
        let cosine = try MLMultiArray(shape: shape, dataType: .float32)
        let sine = try MLMultiArray(shape: shape, dataType: .float32)
        let halfDimension = headDimension / 2
        for position in 0..<sequenceLength {
            for dimension in 0..<headDimension {
                let frequencyIndex = dimension % halfDimension
                let exponent = Double(2 * frequencyIndex) / Double(headDimension)
                let frequency = 1.0 / pow(ropeTheta, exponent)
                let angle = Double(position) * frequency
                let indices = [0, position, dimension].map(NSNumber.init)
                cosine[indices] = NSNumber(value: cos(angle))
                sine[indices] = NSNumber(value: sin(angle))
            }
        }
        return (cosine, sine)
    }

    /// Creates the finite fp16 additive mask `[1,1,S,S]`.
    private static func makeCausalMask(sequenceLength: Int) throws -> MLMultiArray {
        let mask = try MLMultiArray(
            shape: [1, 1, sequenceLength, sequenceLength].map(NSNumber.init),
            dataType: .float16
        )
        for query in 0..<sequenceLength {
            for key in 0..<sequenceLength {
                mask[[0, 0, query, key].map(NSNumber.init)] = NSNumber(
                    value: key > query ? causalMaskNegative : 0.0
                )
            }
        }
        return mask
    }

    /// Compares paired final logits outside the measured wall-time interval.
    private static func maximumAbsoluteDifference(
        _ left: MLMultiArray, _ right: MLMultiArray
    ) throws -> Double {
        guard left.count == right.count else {
            throw RuntimeError.invalidInput("logit output counts differ")
        }
        var maximum = 0.0
        for index in 0..<left.count {
            maximum = max(maximum, abs(left[index].doubleValue - right[index].doubleValue))
        }
        return maximum
    }

    /// Rejects any hidden boundary that Core ML does not materialize as fp16.
    private static func requireFloat16Boundary(
        _ array: MLMultiArray, label: String
    ) throws {
        guard array.dataType == .float16 else {
            throw RuntimeError.invalidInput(
                "\(label) must be float16, found \(dataTypeName(array.dataType))"
            )
        }
    }

    /// Matches the Python evidence hash over sorted relative paths and bytes.
    private static func packageTreeSHA256(_ package: URL) throws -> String {
        guard let enumerator = FileManager.default.enumerator(
            at: package,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw RuntimeError.invalidInput("cannot enumerate package \(package.path)")
        }
        let files = enumerator.compactMap { $0 as? URL }.filter { url in
            (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        }.sorted { $0.path < $1.path }
        var hasher = SHA256()
        let packagePrefix = package.standardizedFileURL.path + "/"
        for file in files {
            let standardizedPath = file.standardizedFileURL.path
            guard standardizedPath.hasPrefix(packagePrefix) else {
                throw RuntimeError.invalidInput("file escaped package root")
            }
            let relativePath = String(standardizedPath.dropFirst(packagePrefix.count))
            hasher.update(data: Data(relativePath.utf8))
            hasher.update(data: Data([0]))
            let handle = try FileHandle(forReadingFrom: file)
            defer { try? handle.close() }
            while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty {
                hasher.update(data: chunk)
            }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Returns the median and inclusive linear-interpolated IQR.
    private static func summary(_ values: [Double]) -> (median: Double, iqr: Double) {
        let sorted = values.sorted()
        let median = percentile(sorted, fraction: 0.5)
        let q1 = percentile(sorted, fraction: 0.25)
        let q3 = percentile(sorted, fraction: 0.75)
        return (median, q3 - q1)
    }

    /// Matches Python's inclusive quantile interpolation for durable parity.
    private static func percentile(_ sorted: [Double], fraction: Double) -> Double {
        let position = fraction * Double(sorted.count - 1)
        let lower = Int(floor(position))
        let upper = Int(ceil(position))
        if lower == upper {
            return sorted[lower]
        }
        let weight = position - Double(lower)
        return sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }

    /// Produces a stable evidence string for the actual boundary storage type.
    private static func dataTypeName(_ dataType: MLMultiArrayDataType) -> String {
        switch dataType {
        case .double: return "double"
        case .float32: return "float32"
        case .float16: return "float16"
        case .int32: return "int32"
        case .int8: return "int8"
        @unknown default: return "unknown"
        }
    }
}

/// Fail-closed runtime errors with direct experiment context.
public enum RuntimeError: Error, CustomStringConvertible {
    /// Invalid frozen input or benchmark count.
    case invalidInput(String)

    /// A Core ML model did not produce its documented output.
    case missingOutput(String)

    /// Human-readable failure context for CLI stderr.
    public var description: String {
        switch self {
        case .invalidInput(let message): return message
        case .missingOutput(let name): return "Core ML prediction omitted \(name)"
        }
    }
}
