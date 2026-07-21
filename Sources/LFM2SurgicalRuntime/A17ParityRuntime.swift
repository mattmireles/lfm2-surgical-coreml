import CoreML
import Darwin
import Foundation

/// One hash-bound same-phone compute-plan reference approved by the host gate.
public struct A17ParityDispatchReference: Codable, Sendable {
    /// Requested Core ML policy (`CPU_AND_GPU` or `CPU_AND_NE`).
    public let computeUnits: String

    /// SHA-256 of the canonical dispatch JSON file.
    public let evidenceSha256: String

    /// Relative durable path under the A17 evidence root.
    public let path: String

    /// Exact fixed-twin package-tree SHA-256 described by the plan.
    public let packageSha256: String
}

/// One source package copied into the app from the frozen Stage 1 artifacts.
public struct A17ParityManifestArtifact: Codable, Sendable {
    /// Stable artifact label used by the runtime and result validator.
    public let name: String

    /// Package path relative to the public repository root.
    public let path: String

    /// Xcode-compiled model stem available in the application bundle.
    public let resourceName: String

    /// SHA-256 over the source package tree before Xcode compilation.
    public let sha256: String
}

/// Build-time contract written only by `ios-a17-parity/prepare_resources.sh`.
public struct A17ParityManifest: Codable, Sendable {
    /// Version of the manifest contract.
    public let schemaVersion: Int

    /// Git revision of the public app/runtime sources.
    public let appRevision: String

    /// Frozen CoreDevice identifier revalidated by the host around device I/O.
    public let deviceIdentifier: String

    /// Frozen physical hardware UDID revalidated by the host.
    public let hardwareUDID: String

    /// Frozen physical product type (`iPhone16,2`).
    public let productType: String

    /// Exactly one registered convolution-pair name.
    public let pair: String

    /// Xcode-compiled resource stem for the fixed-512 twin.
    public let fixedPairResourceName: String

    /// Bundled Stage 1 input JSON resource stem.
    public let inputResourceName: String

    /// Pair-specific result filename written under the app Documents directory.
    public let resultFilename: String

    /// Source-package identities verified before and after staging.
    public let artifacts: [A17ParityManifestArtifact]

    /// Two validated same-phone dispatch references for the fixed twin.
    public let dispatch: [A17ParityDispatchReference]
}

/// Device and OS identity observed by the application at prediction time.
public struct A17ParityDeviceEvidence: Codable, Sendable {
    /// Host-validated CoreDevice identifier carried by the signed manifest.
    public let coredeviceIdentifier: String

    /// Host-validated hardware UDID carried by the signed manifest.
    public let hardwareUDID: String

    /// Product type observed locally through `uname`.
    public let productType: String

    /// Public operating-system version observed in the app process.
    public let osVersion: String

    /// Operating-system build parsed from `ProcessInfo` when available.
    public let osBuild: String
}

/// Exact signed app identity that generated a pair result.
public struct A17ParityAppEvidence: Codable, Sendable {
    /// Bundle identifier from the signed application.
    public let bundleIdentifier: String

    /// Build number from the signed application Info.plist.
    public let buildNumber: String

    /// Git revision written into the immutable resource manifest.
    public let revision: String
}

/// Frozen all-GPU oracle result produced before candidate comparison.
public struct A17ParityOracleEvidence: Codable, Sendable {
    /// Required top token from the complete source-segment path.
    public let topToken: Int

    /// True only for the preregistered token `941`.
    public let passed: Bool
}

/// One first-use or warmed correctness comparison with no timing fields.
public struct A17ParityComparison: Codable, Sendable {
    /// Zero-based row index; the complete result requires exactly six rows.
    public let repetition: Int

    /// `first_use` for row zero and `warmed` for rows one through five.
    public let repetitionKind: String

    /// Candidate call order (`GPU->ANE` or `ANE->GPU`).
    public let candidateOrder: String

    /// Per-output error between the source package and fixed GPU twin.
    public let sourceTwinMaxAbsByOutput: [String: Double]

    /// Largest source-versus-twin named-output error.
    public let sourceTwinMaxAbs: Double

    /// Per-output error between fixed GPU and fixed ANE-permitted execution.
    public let gpuAneMaxAbsByOutput: [String: Double]

    /// Largest GPU-versus-ANE named-output error.
    public let pairOutputMaxAbs: Double

    /// Final-logit error after independent copies of the common GPU suffix.
    public let suffixLogitMaxAbs: Double

    /// Frozen complete-source oracle token.
    public let oracleToken: Int

    /// Final token after the fixed GPU pair and common suffix.
    public let gpuToken: Int

    /// Final token after the fixed ANE-permitted pair and common suffix.
    public let aneToken: Int

    /// True only when every numerical and token gate passes.
    public let passed: Bool
}

/// Versioned, untimed result produced by one registered A17 pair probe.
public struct A17ParityResult: Codable, Sendable {
    /// Version of the result contract.
    public let schemaVersion: Int

    /// ISO-8601 completion timestamp; never used as a performance sample.
    public let capturedAtUTC: String

    /// Runtime identity proving direct Swift/Core ML execution.
    public let runtime: String

    /// Live device/OS identity plus host-validated private identifiers.
    public let device: A17ParityDeviceEvidence

    /// Signed application identity.
    public let app: A17ParityAppEvidence

    /// Exact frozen checkpoint identity from the Stage 1 input.
    public let checkpoint: CheckpointIdentity

    /// Registered prompt bucket; this probe accepts only 512.
    public let bucket: Int

    /// Exact registered prompt.
    public let prompt: String

    /// Frozen model/tensor precision label.
    public let precision: String

    /// Frozen maximum-absolute numerical tolerance.
    public let maximumAbsoluteTolerance: Double

    /// Exactly one registered pair name.
    public let pair: String

    /// Source package hashes copied into this app build.
    public let artifacts: [A17ParityManifestArtifact]

    /// Hash-bound GPU and ANE compute plans approved before launch.
    public let dispatch: [A17ParityDispatchReference]

    /// Complete all-GPU oracle result.
    public let oracle: A17ParityOracleEvidence

    /// One first-use plus exactly five warmed correctness rows.
    public let comparisons: [A17ParityComparison]

    /// `PAIR PASS` or `KILL`; never a performance verdict.
    public let verdict: String

    /// Frozen proof that this schema records no performance samples.
    public let performanceSamplesRecorded: Int
}

/// Deterministic, untimed physical-iPhone parity runner for Plan 012.
public final class A17ParityRuntime {
    /// Frozen bucket-specific correctness threshold.
    private static let maximumAbsoluteTolerance = 1.0e-2

    /// Frozen complete-GPU token for the registered prompt.
    private static let expectedOracleToken = 941

    /// One first-use order followed by the five registered warmed orders.
    private static let comparisonOrders = [
        "GPU->ANE",
        "GPU->ANE",
        "ANE->GPU",
        "GPU->ANE",
        "ANE->GPU",
        "GPU->ANE",
    ]

    /// Phase-order labels; definitions remain owned by `SelectivePrefillRuntime`.
    public static let registeredPairOrder = [
        "segment_04_conv_6_7",
        "segment_00_conv_0_1",
        "segment_02_conv_3_4",
    ]

    /// Signed build-time manifest.
    private let manifest: A17ParityManifest

    /// Exact Stage 1 input bundled with the app.
    private let input: Stage1Input

    /// Common GPU-permitted embedding.
    private let embedding: MLModel

    /// Common GPU-permitted final norm and tied head.
    private let languageModelHead: MLModel

    /// Complete frozen source-segment chain under `.cpuAndGPU`.
    private let oracleSegments: [(SegmentDescriptor, MLModel)]

    /// Registered pair metadata shared with the selective runtime.
    private let pairDefinition: PairDefinition

    /// Fixed twin loaded with ANE excluded.
    private let fixedGPU: MLModel

    /// Identical fixed twin loaded with ANE permitted.
    private let fixedANE: MLModel

    /// Exact embedding input.
    private let tokenIDs: MLMultiArray

    /// Exact fp16 embedding mask.
    private let tokenMask: MLMultiArray

    /// Frozen fp32 rotary cosine table.
    private let cosine: MLMultiArray

    /// Frozen fp32 rotary sine table.
    private let sine: MLMultiArray

    /// Frozen fp16 causal mask.
    private let attentionMask: MLMultiArray

    /// Zero-valued explicit prefill convolution state.
    private let zeroState: MLMultiArray

    /// Loads the immutable manifest and every required compiled resource.
    public init(resourceBundle: Bundle = .main) throws {
        guard let manifestURL = resourceBundle.url(
            forResource: "a17-parity-manifest", withExtension: "json"
        ) else {
            throw RuntimeError.invalidInput("missing a17-parity-manifest.json")
        }
        let decodedManifest = try JSONDecoder().decode(
            A17ParityManifest.self, from: Data(contentsOf: manifestURL)
        )
        guard decodedManifest.schemaVersion == 1 else {
            throw RuntimeError.invalidInput("unsupported A17 manifest schema")
        }
        guard Self.registeredPairOrder.contains(decodedManifest.pair),
              let definition = SelectivePrefillRuntime.pairDefinitions.first(
                where: { $0.name == decodedManifest.pair }
              ) else {
            throw RuntimeError.invalidInput("unregistered A17 pair")
        }
        manifest = decodedManifest
        pairDefinition = definition
        guard let inputURL = resourceBundle.url(
            forResource: manifest.inputResourceName, withExtension: "json"
        ) else {
            throw RuntimeError.invalidInput("missing frozen Stage 1 input")
        }
        input = try JSONDecoder().decode(
            Stage1Input.self, from: Data(contentsOf: inputURL)
        )
        guard input.bucket == 512,
              input.tokenIds.count == input.bucket,
              input.prompt
                == "Explain why fixed tensor shapes can help a compiler target an accelerator." else {
            throw RuntimeError.invalidInput("A17 input contract drifted")
        }

        let gpu = MLModelConfiguration()
        gpu.computeUnits = .cpuAndGPU
        let ane = MLModelConfiguration()
        ane.computeUnits = .cpuAndNeuralEngine
        embedding = try Self.loadCompiledModel(
            named: "embedding", bundle: resourceBundle, configuration: gpu
        )
        languageModelHead = try Self.loadCompiledModel(
            named: "lm_head", bundle: resourceBundle, configuration: gpu
        )
        oracleSegments = try SurgicalPrefillRuntime.segments.map { descriptor in
            (
                descriptor,
                try Self.loadCompiledModel(
                    named: descriptor.name,
                    bundle: resourceBundle,
                    configuration: gpu
                )
            )
        }
        fixedGPU = try Self.loadCompiledModel(
            named: manifest.fixedPairResourceName,
            bundle: resourceBundle,
            configuration: gpu
        )
        fixedANE = try Self.loadCompiledModel(
            named: manifest.fixedPairResourceName,
            bundle: resourceBundle,
            configuration: ane
        )
        tokenIDs = try SurgicalPrefillRuntime.makeTokenIDs(input.tokenIds)
        tokenMask = try SurgicalPrefillRuntime.makeFilledArray(
            shape: [1, 1, 1, NSNumber(value: input.bucket)],
            dataType: .float16,
            value: 1.0
        )
        (cosine, sine) = try SurgicalPrefillRuntime.makeRoPETables(
            sequenceLength: input.bucket
        )
        attentionMask = try SurgicalPrefillRuntime.makeCausalMask(
            sequenceLength: input.bucket
        )
        zeroState = try SurgicalPrefillRuntime.makeFilledArray(
            shape: [
                1,
                SurgicalPrefillRuntime.hiddenSize,
                SurgicalPrefillRuntime.convolutionStateWidth,
            ].map(NSNumber.init),
            dataType: .float16,
            value: 0.0
        )
        try validateManifest()
    }

    /// Runs one first-use and five warmed correctness comparisons without clocks.
    public func run() throws -> A17ParityResult {
        let oracle = try runOracle()
        var comparisons: [A17ParityComparison] = []
        comparisons.reserveCapacity(Self.comparisonOrders.count)
        for (index, order) in Self.comparisonOrders.enumerated() {
            comparisons.append(
                try runComparison(
                    repetition: index,
                    kind: index == 0 ? "first_use" : "warmed",
                    order: order,
                    entry: oracle.entry,
                    oracleToken: oracle.token
                )
            )
        }
        let passed = oracle.token == Self.expectedOracleToken
            && comparisons.count == Self.comparisonOrders.count
            && comparisons.allSatisfy(\.passed)
        return A17ParityResult(
            schemaVersion: 1,
            capturedAtUTC: ISO8601DateFormatter().string(from: Date()),
            runtime: "Release Swift/Core ML; untimed A17 parity probe",
            device: Self.deviceEvidence(manifest: manifest),
            app: A17ParityAppEvidence(
                bundleIdentifier: Bundle.main.bundleIdentifier ?? "unknown",
                buildNumber: Bundle.main.object(
                    forInfoDictionaryKey: "CFBundleVersion"
                ) as? String ?? "unknown",
                revision: manifest.appRevision
            ),
            checkpoint: input.checkpoint,
            bucket: input.bucket,
            prompt: input.prompt,
            precision: "fp16",
            maximumAbsoluteTolerance: Self.maximumAbsoluteTolerance,
            pair: manifest.pair,
            artifacts: manifest.artifacts,
            dispatch: manifest.dispatch,
            oracle: A17ParityOracleEvidence(
                topToken: oracle.token,
                passed: oracle.token == Self.expectedOracleToken
            ),
            comparisons: comparisons,
            verdict: passed ? "PAIR PASS" : "KILL",
            performanceSamplesRecorded: 0
        )
    }

    /// Pair-specific durable filename selected by the signed manifest.
    public var resultFilename: String {
        manifest.resultFilename
    }

    /// Rejects build-time drift before any prediction occurs.
    private func validateManifest() throws {
        guard manifest.deviceIdentifier
                == "8A12AEE8-0136-50BE-8EB3-91650E467F15",
              manifest.hardwareUDID == "00008130-000600660EA2001C",
              manifest.productType == "iPhone16,2" else {
            throw RuntimeError.invalidInput("A17 device manifest drifted")
        }
        guard manifest.dispatch.count == 2,
              Set(manifest.dispatch.map(\.computeUnits))
                == Set(["CPU_AND_GPU", "CPU_AND_NE"]),
              Set(manifest.dispatch.map(\.packageSha256)).count == 1 else {
            throw RuntimeError.invalidInput("A17 dispatch manifest drifted")
        }
        guard let fixed = manifest.artifacts.first(
            where: { $0.name == "\(manifest.pair)_fixed512" }
        ), fixed.resourceName == manifest.fixedPairResourceName,
           fixed.sha256 == manifest.dispatch[0].packageSha256 else {
            throw RuntimeError.invalidInput("A17 fixed-pair hash drifted")
        }
        guard Set(manifest.artifacts.map(\.name)).count == manifest.artifacts.count else {
            throw RuntimeError.invalidInput("A17 artifact names are not unique")
        }
    }

    /// Runs the complete source chain and captures the real registered pair entry.
    private func runOracle() throws -> (entry: MLMultiArray, token: Int) {
        var hidden = try SurgicalPrefillRuntime.predict(
            model: embedding,
            values: [
                "token_ids": try SurgicalPrefillRuntime.copyMultiArray(tokenIDs),
                "token_mask": try SurgicalPrefillRuntime.copyMultiArray(tokenMask),
            ],
            output: "hidden_out"
        )
        try SurgicalPrefillRuntime.requireFloat16Boundary(
            hidden, label: "A17 embedding hidden output"
        )
        var entry: MLMultiArray?
        for (index, segment) in oracleSegments.enumerated() {
            if index == pairDefinition.segmentIndex {
                entry = try SurgicalPrefillRuntime.copyMultiArray(hidden)
            }
            hidden = try runSegment(
                descriptor: segment.0, model: segment.1, hidden: hidden
            )
        }
        guard let pairEntry = entry else {
            throw RuntimeError.invalidInput("A17 oracle did not capture pair entry")
        }
        let logits = try SurgicalPrefillRuntime.runHead(
            model: languageModelHead,
            hidden: try SurgicalPrefillRuntime.copyMultiArray(hidden),
            lastPosition: input.bucket - 1
        )
        return (pairEntry, Self.topToken(logits))
    }

    /// Produces one independent source/GPU/ANE comparison and suffix validation.
    private func runComparison(
        repetition: Int,
        kind: String,
        order: String,
        entry: MLMultiArray,
        oracleToken: Int
    ) throws -> A17ParityComparison {
        let source = try pairOutputs(
            model: oracleSegments[pairDefinition.segmentIndex].1,
            hidden: entry
        )
        var candidates: [String: [String: MLMultiArray]] = [:]
        for name in order.split(separator: "->").map(String.init) {
            candidates[name] = try pairOutputs(
                model: name == "GPU" ? fixedGPU : fixedANE,
                hidden: entry
            )
        }
        guard let gpu = candidates["GPU"], let ane = candidates["ANE"] else {
            throw RuntimeError.invalidInput("A17 candidate order lost one policy")
        }
        let sourceErrors = try outputDifferences(source, gpu)
        let candidateErrors = try outputDifferences(gpu, ane)
        let gpuLogits = try suffixLogits(
            hidden: gpu["hidden_out"]!, after: pairDefinition.segmentIndex
        )
        let aneLogits = try suffixLogits(
            hidden: ane["hidden_out"]!, after: pairDefinition.segmentIndex
        )
        let suffixError = try SurgicalPrefillRuntime.maximumAbsoluteDifference(
            gpuLogits, aneLogits
        )
        let gpuToken = Self.topToken(gpuLogits)
        let aneToken = Self.topToken(aneLogits)
        let sourceMaximum = sourceErrors.values.max() ?? .infinity
        let candidateMaximum = candidateErrors.values.max() ?? .infinity
        let passed = sourceMaximum <= Self.maximumAbsoluteTolerance
            && candidateMaximum <= Self.maximumAbsoluteTolerance
            && suffixError <= Self.maximumAbsoluteTolerance
            && oracleToken == Self.expectedOracleToken
            && gpuToken == Self.expectedOracleToken
            && aneToken == Self.expectedOracleToken
        return A17ParityComparison(
            repetition: repetition,
            repetitionKind: kind,
            candidateOrder: order,
            sourceTwinMaxAbsByOutput: sourceErrors,
            sourceTwinMaxAbs: sourceMaximum,
            gpuAneMaxAbsByOutput: candidateErrors,
            pairOutputMaxAbs: candidateMaximum,
            suffixLogitMaxAbs: suffixError,
            oracleToken: oracleToken,
            gpuToken: gpuToken,
            aneToken: aneToken,
            passed: passed
        )
    }

    /// Runs one pair and snapshots all required named outputs before returning.
    private func pairOutputs(
        model: MLModel, hidden: MLMultiArray
    ) throws -> [String: MLMultiArray] {
        var values: [String: MLMultiArray] = [
            "hidden_states": try SurgicalPrefillRuntime.copyMultiArray(hidden)
        ]
        for name in pairDefinition.stateInputs {
            values[name] = try SurgicalPrefillRuntime.copyMultiArray(zeroState)
        }
        let prediction = try SurgicalPrefillRuntime.predictAll(
            model: model, values: values
        )
        return try Dictionary(
            uniqueKeysWithValues: pairDefinition.outputNames.map { name in
                guard let output = prediction.featureValue(
                    for: name
                )?.multiArrayValue else {
                    throw RuntimeError.missingOutput(name)
                }
                return (name, try SurgicalPrefillRuntime.copyMultiArray(output))
            }
        )
    }

    /// Runs an independent copy of the common GPU suffix and head.
    private func suffixLogits(
        hidden initialHidden: MLMultiArray, after segmentIndex: Int
    ) throws -> MLMultiArray {
        var hidden = try SurgicalPrefillRuntime.copyMultiArray(initialHidden)
        if segmentIndex + 1 < oracleSegments.count {
            for index in (segmentIndex + 1)..<oracleSegments.count {
                hidden = try runSegment(
                    descriptor: oracleSegments[index].0,
                    model: oracleSegments[index].1,
                    hidden: hidden
                )
            }
        }
        let logits = try SurgicalPrefillRuntime.runHead(
            model: languageModelHead,
            hidden: try SurgicalPrefillRuntime.copyMultiArray(hidden),
            lastPosition: input.bucket - 1
        )
        return try SurgicalPrefillRuntime.copyMultiArray(logits)
    }

    /// Runs one source segment from independently copied inputs and state.
    private func runSegment(
        descriptor: SegmentDescriptor,
        model: MLModel,
        hidden: MLMultiArray
    ) throws -> MLMultiArray {
        var values: [String: MLMultiArray] = [
            "hidden_states": try SurgicalPrefillRuntime.copyMultiArray(hidden)
        ]
        if descriptor.usesAttentionInputs {
            values["cosine"] = try SurgicalPrefillRuntime.copyMultiArray(cosine)
            values["sine"] = try SurgicalPrefillRuntime.copyMultiArray(sine)
            values["attention_mask"] = try SurgicalPrefillRuntime.copyMultiArray(
                attentionMask
            )
        } else {
            for name in descriptor.stateInputs {
                values[name] = try SurgicalPrefillRuntime.copyMultiArray(zeroState)
            }
        }
        let output = try SurgicalPrefillRuntime.predict(
            model: model, values: values, output: "hidden_out"
        )
        try SurgicalPrefillRuntime.requireFloat16Boundary(
            output, label: "\(descriptor.name) hidden output"
        )
        return try SurgicalPrefillRuntime.copyMultiArray(output)
    }

    /// Computes exact per-name errors and rejects missing or extra outputs.
    private func outputDifferences(
        _ baseline: [String: MLMultiArray],
        _ candidate: [String: MLMultiArray]
    ) throws -> [String: Double] {
        guard Set(baseline.keys) == Set(pairDefinition.outputNames),
              Set(candidate.keys) == Set(pairDefinition.outputNames) else {
            throw RuntimeError.invalidInput("A17 pair output names drifted")
        }
        return try Dictionary(
            uniqueKeysWithValues: pairDefinition.outputNames.map { name in
                (
                    name,
                    try SurgicalPrefillRuntime.maximumAbsoluteDifference(
                        baseline[name]!, candidate[name]!
                    )
                )
            }
        )
    }

    /// Loads one Xcode-compiled model by its unambiguous bundle stem.
    private static func loadCompiledModel(
        named name: String,
        bundle: Bundle,
        configuration: MLModelConfiguration
    ) throws -> MLModel {
        guard let url = bundle.url(forResource: name, withExtension: "mlmodelc") else {
            throw RuntimeError.invalidInput("missing compiled model \(name).mlmodelc")
        }
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    /// Returns the maximum finite logit index without allocating an array.
    private static func topToken(_ logits: MLMultiArray) -> Int {
        var bestIndex = 0
        var bestValue = -Double.infinity
        for index in 0..<logits.count {
            let value = logits[index].doubleValue
            if value > bestValue {
                bestValue = value
                bestIndex = index
            }
        }
        return bestIndex
    }

    /// Captures public live OS/product data and manifest-bound private identity.
    private static func deviceEvidence(
        manifest: A17ParityManifest
    ) -> A17ParityDeviceEvidence {
        var system = utsname()
        uname(&system)
        let productType = withUnsafePointer(to: &system.machine) { pointer in
            pointer.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(cString: $0)
            }
        }
        let version = ProcessInfo.processInfo.operatingSystemVersion
        let osVersion = "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
        let description = ProcessInfo.processInfo.operatingSystemVersionString
        let osBuild: String
        if let marker = description.range(of: "Build ") {
            osBuild = String(
                description[marker.upperBound...].prefix { $0 != ")" }
            )
        } else {
            osBuild = description
        }
        return A17ParityDeviceEvidence(
            coredeviceIdentifier: manifest.deviceIdentifier,
            hardwareUDID: manifest.hardwareUDID,
            productType: productType,
            osVersion: osVersion,
            osBuild: osBuild
        )
    }
}
