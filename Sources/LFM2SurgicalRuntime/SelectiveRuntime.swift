import CoreML
import CryptoKit
import Darwin
import Foundation

/// One non-benchmark process above the frozen CPU-noise threshold.
public struct HostProcessNoise: Codable, Sendable {
    /// Process identifier from the preflight snapshot.
    public let pid: Int32

    /// Instantaneous process CPU percentage reported by `ps`.
    public let cpuPercent: Double

    /// Executable path or process name reported by `ps`.
    public let command: String
}

/// Durable quiet-host evidence captured immediately before any warmup.
public struct HostNoiseEvidence: Codable, Sendable {
    /// ISO-8601 timestamp for the preflight snapshot.
    public let capturedAtUTC: String

    /// Logical processor count used to normalize load average.
    public let logicalProcessorCount: Int

    /// One-, five-, and fifteen-minute load averages.
    public let loadAverages: [Double]

    /// One-minute load divided by logical processor count.
    public let oneMinuteLoadPerProcessor: Double

    /// System-wide free-memory percentage from `memory_pressure -Q`.
    public let memoryFreePercent: Int

    /// Allocated swap in mebibytes; recorded but not used as a kill threshold.
    public let swapUsedMiB: Double

    /// Other processes at or above 50% CPU when the snapshot was taken.
    public let highCPUProcesses: [HostProcessNoise]

    /// Frozen preflight rule and thresholds.
    public let policy: String

    /// True when load, memory pressure, and competing-process gates pass.
    public let passed: Bool
}

/// One candidate's raw and summarized warmed latency samples.
public struct SelectiveCandidateTiming: Codable, Sendable {
    /// Frozen candidate label from plan 011.
    public let name: String

    /// Requested Core ML compute-unit policy.
    public let computeUnits: String

    /// Package-tree SHA-256 shared by the paired candidates.
    public let packageSha256: String

    /// One elapsed wall-clock value per counterbalanced pair.
    public let timesMs: [Double]

    /// Inclusive sample median, or `nil` when correctness cancels timing.
    public let medianMs: Double?

    /// Inclusive interquartile range, or `nil` when timing is cancelled.
    public let iqrMs: Double?
}

/// Deterministic paired percentile-bootstrap evidence.
public struct PairedBootstrapEvidence: Codable, Sendable {
    /// Positive values favor the candidate over the baseline.
    public let signedDeltaDefinition: String

    /// Baseline minus candidate for every paired observation.
    public let pairedDeltasMs: [Double]

    /// Difference between the two independently summarized medians.
    public let pointEstimateMs: Double

    /// Lower type-7 percentile bound.
    public let lower95Ms: Double

    /// Upper type-7 percentile bound.
    public let upper95Ms: Double

    /// Number of paired bootstrap resamples.
    public let resamples: Int

    /// Deterministic SplitMix64 seed.
    public let seed: UInt64
}

/// Hash-bound same-host compute-plan summary used by a timing row.
public struct SelectiveDispatchEvidence: Codable, Sendable {
    /// Stable candidate-policy label.
    public let name: String

    /// JSON path relative to the dispatch directory's parent.
    public let path: String

    /// SHA-256 of the compute-plan JSON itself.
    public let evidenceSha256: String

    /// SHA-256 of the exact package described by the plan.
    public let packageSha256: String

    /// Requested compute-unit permission.
    public let computeUnits: String

    /// Number of operations with a preferred compute device.
    public let totalCostedOps: Int

    /// Preferred-device histogram copied from the plan.
    public let preferredDeviceHistogram: [String: Int]

    /// Per-operator preferred-device histogram copied from the plan.
    public let perOpType: [String: [String: Int]]
}

/// The zero-public-boundary scheduler comparison and G0a decision.
public struct SelectiveMonolithComparison: Codable, Sendable {
    /// `.cpuAndGPU` no-boundary control.
    public let mGPU: SelectiveCandidateTiming

    /// Same package loaded with `.all`.
    public let mALL: SelectiveCandidateTiming

    /// Alternating candidate order for every measurement pair.
    public let runOrder: [String]

    /// Paired evidence for `M_GPU - M_ALL`.
    public let improvement: PairedBootstrapEvidence

    /// Maximum final-logit error across all pairs.
    public let outputMaxAbs: Double

    /// Argmax token from the GPU control.
    public let mGPUTopToken: Int

    /// Argmax token from the `.all` scheduler candidate.
    public let mALLTopToken: Int

    /// True when the monolith places a broader conv set on ANE and attention
    /// defining matmul/softmax operations on GPU.
    public let heterogeneousDispatch: Bool

    /// True when `.all` is no slower in median and its interval is not wholly
    /// on the regression side of zero.
    public let noSupportedRegression: Bool

    /// True when final top token and the frozen `1e-2` tolerance pass.
    public let numericsPassed: Bool

    /// Frozen G0a decision.
    public let passedG0a: Bool
}

/// One fixed-512 conv pair under the two registered placement policies.
public struct SelectivePairComparison: Codable, Sendable {
    /// Frozen segment label.
    public let name: String

    /// Enumerated source package from plan 010.
    public let sourcePackageSha256: String

    /// Source-versus-materialized maximum error over hidden and both states.
    public let sourceTwinMaxAbs: Double

    /// Fixed twin under `.cpuAndGPU`.
    public let gpu: SelectiveCandidateTiming

    /// Same fixed twin under `.cpuAndNeuralEngine`.
    public let ane: SelectiveCandidateTiming

    /// Alternating GPU/ANE order for every measurement pair.
    public let runOrder: [String]

    /// Paired evidence for `GPU median - ANE median`, absent when untimed.
    public let improvement: PairedBootstrapEvidence?

    /// Maximum error over hidden and both state outputs.
    public let outputMaxAbs: Double

    /// Maximum final-logit error after the common untimed GPU suffix.
    public let finalLogitMaxAbs: Double

    /// Final argmax after the GPU pair and common suffix.
    public let gpuTopToken: Int

    /// Final argmax after the ANE pair and common suffix.
    public let aneTopToken: Int

    /// True only when source materialization and placement numerics pass.
    public let numericsPassed: Bool
}

/// Explicit outcomes for the four frozen Phase 0 gates.
public struct SelectiveG0Outcomes: Codable, Sendable {
    public let g0a: Bool
    public let g0b: Bool?
    public let g0c: Bool?
    public let g0d: Bool?
}

/// Durable Phase 0 evidence written before the process chooses continuation.
public struct SelectivePhase0Result: Codable, Sendable {
    /// Schema version for future device commands.
    public let schemaVersion: Int

    /// Runtime identity proving direct Swift/Core ML boundaries.
    public let runtime: String

    /// Frozen checkpoint identity from the exact Stage 1 input.
    public let checkpoint: CheckpointIdentity

    /// Host and conversion-tool provenance.
    public let environment: RuntimeEnvironment

    /// Exact unique packages used by the full-path comparison.
    public let artifacts: [ArtifactIdentity]

    /// Hash-bound same-host compute plans.
    public let dispatch: [SelectiveDispatchEvidence]

    /// Frozen prompt bucket.
    public let bucket: Int

    /// Registered real prompt.
    public let prompt: String

    /// Number of untimed warmups per candidate.
    public let warmupsPerCandidate: Int

    /// Number of measured pairs per candidate.
    public let measuredRunsPerCandidate: Int

    /// Host thermal state before any warmed measurement.
    public let thermalStateBefore: String

    /// Load, memory-pressure, swap, and competing-process preflight evidence.
    public let hostNoiseBefore: HostNoiseEvidence

    /// Host thermal state after the comparison.
    public let thermalStateAfter: String

    /// Full no-boundary scheduler comparison.
    public let monolith: SelectiveMonolithComparison

    /// Empty when G0a or G0b terminates before pair correctness checks.
    public let pairs: [SelectivePairComparison]

    /// Aggregate paired savings across the three pairs when timing is valid.
    public let aggregatePairSavings: PairedBootstrapEvidence?

    /// Machine-readable outcomes for G0a through G0d.
    public let gates: SelectiveG0Outcomes

    /// First gate that terminates the experiment, or `nil` when Phase 0 passes.
    public let terminalGate: String?

    /// `NO SPLIT NEEDED`, `KILL`, or `GO`.
    public let verdict: String

    /// True only when all Phase 0 gates authorize the next plan phase.
    public let continuePhase0: Bool

    /// Explicitly records why pair timing is absent from a G0a terminal result.
    public let cancelledMeasurements: [String]
}

/// Additive plan-011 runtime. The frozen 13-segment executable remains a
/// separate public API and is replayed after this phase.
public final class SelectivePrefillRuntime {
    /// Plan 011 freezes this exact sample count.
    private static let requiredWarmups = 3

    /// Plan 011 freezes this exact measured pair count.
    private static let requiredRuns = 20

    /// Frozen fp16 final-logit tolerance.
    private static let maximumLogitError = 1.0e-2

    /// Sufficient precision for the registered paired confidence interval.
    private static let bootstrapResamples = 100_000

    /// Recorded deterministic seed; changing it changes durable evidence.
    private static let bootstrapSeed: UInt64 = 0x4C46_4D32_5048_3041

    /// Exactly the three pre-registered two-convolution islands.
    static let pairDefinitions = [
        PairDefinition(
            name: "segment_00_conv_0_1",
            segmentIndex: 0,
            stateInputs: ["conv_state_0_in", "conv_state_1_in"],
            outputNames: ["hidden_out", "conv_state_0_out", "conv_state_1_out"]
        ),
        PairDefinition(
            name: "segment_02_conv_3_4",
            segmentIndex: 2,
            stateInputs: ["conv_state_3_in", "conv_state_4_in"],
            outputNames: ["hidden_out", "conv_state_3_out", "conv_state_4_out"]
        ),
        PairDefinition(
            name: "segment_04_conv_6_7",
            segmentIndex: 4,
            stateInputs: ["conv_state_6_in", "conv_state_7_in"],
            outputNames: ["hidden_out", "conv_state_6_out", "conv_state_7_out"]
        ),
    ]

    /// Exact prompt and checkpoint identity.
    private let input: Stage1Input

    /// Root of the frozen Stage 1 packages.
    private let stage1Directory: URL

    /// Directory containing same-host plan JSON files.
    private let dispatchDirectory: URL

    /// Root containing the three amended fixed-512 pair twins.
    private let fixedPairDirectory: URL

    /// Common embedding, always GPU-permitted.
    private let embedding: MLModel

    /// Common final norm and tied head, always GPU-permitted.
    private let languageModelHead: MLModel

    /// Fixed-512 monolith with ANE excluded.
    private let monolithGPU: MLModel

    /// Identical fixed-512 monolith with all devices permitted.
    private let monolithALL: MLModel

    /// Reusable exact input tensors.
    private let tokenIDs: MLMultiArray
    private let tokenMask: MLMultiArray
    private let cosine: MLMultiArray
    private let sine: MLMultiArray
    private let attentionMask: MLMultiArray
    private let zeroState: MLMultiArray

    /// Compiles and loads every Phase 0 G0a model before timing.
    public init(
        stage1Directory: URL,
        inputURL: URL,
        dispatchDirectory: URL,
        fixedPairDirectory: URL
    ) throws {
        input = try JSONDecoder().decode(Stage1Input.self, from: Data(contentsOf: inputURL))
        self.stage1Directory = stage1Directory
        self.dispatchDirectory = dispatchDirectory
        self.fixedPairDirectory = fixedPairDirectory
        guard input.bucket == 512, input.tokenIds.count == input.bucket else {
            throw RuntimeError.invalidInput(
                "selective Phase 0 requires exactly 512 frozen token IDs"
            )
        }

        let gpu = MLModelConfiguration()
        gpu.computeUnits = .cpuAndGPU
        let all = MLModelConfiguration()
        all.computeUnits = .all
        embedding = try SurgicalPrefillRuntime.loadModel(
            package: stage1Directory.appendingPathComponent("embedding.mlpackage"),
            configuration: gpu
        )
        languageModelHead = try SurgicalPrefillRuntime.loadModel(
            package: stage1Directory.appendingPathComponent("lm_head.mlpackage"),
            configuration: gpu
        )
        let monolithPackage = stage1Directory.appendingPathComponent(
            "monolithic_layer_stack_512.mlpackage"
        )
        monolithGPU = try SurgicalPrefillRuntime.loadModel(
            package: monolithPackage, configuration: gpu
        )
        monolithALL = try SurgicalPrefillRuntime.loadModel(
            package: monolithPackage, configuration: all
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
    }

    /// Runs G0a. Pair timing remains unexecuted when this gate terminates.
    public func benchmarkPhase0(warmups: Int, runs: Int) throws -> SelectivePhase0Result {
        guard warmups == Self.requiredWarmups, runs == Self.requiredRuns else {
            throw RuntimeError.invalidInput(
                "Phase 0 requires exactly 3 warmups and 20 measured pairs"
            )
        }
        let artifacts = try artifactIdentities()
        let dispatch = try loadAndValidateDispatch(artifacts: artifacts)
        let thermalBefore = Self.thermalStateName(ProcessInfo.processInfo.thermalState)
        guard ProcessInfo.processInfo.thermalState != .serious,
              ProcessInfo.processInfo.thermalState != .critical else {
            throw RuntimeError.invalidInput(
                "refusing Phase 0 timing under \(thermalBefore) thermal state"
            )
        }
        let hostNoiseBefore = try Self.captureHostNoiseEvidence()
        guard hostNoiseBefore.passed else {
            throw RuntimeError.invalidInput(
                "refusing Phase 0 timing because the host-noise preflight failed"
            )
        }

        for _ in 0..<warmups {
            _ = try runMonolith(monolithGPU)
            _ = try runMonolith(monolithALL)
        }

        var gpuTimes: [Double] = []
        var allTimes: [Double] = []
        var runOrder: [String] = []
        var outputMaxAbs = 0.0
        var gpuTopToken = -1
        var allTopToken = -1
        for runIndex in 0..<runs {
            let order = runIndex.isMultiple(of: 2)
                ? ["M_GPU", "M_ALL"]
                : ["M_ALL", "M_GPU"]
            runOrder.append(order.joined(separator: "->"))
            var outputs: [String: MLMultiArray] = [:]
            for candidate in order {
                let model = candidate == "M_GPU" ? monolithGPU : monolithALL
                let start = DispatchTime.now().uptimeNanoseconds
                let logits = try runMonolith(model)
                let elapsed = Double(
                    DispatchTime.now().uptimeNanoseconds - start
                ) / 1_000_000.0
                outputs[candidate] = logits
                if candidate == "M_GPU" {
                    gpuTimes.append(elapsed)
                } else {
                    allTimes.append(elapsed)
                }
            }
            let gpuOutput = outputs["M_GPU"]!
            let allOutput = outputs["M_ALL"]!
            outputMaxAbs = max(
                outputMaxAbs,
                try SurgicalPrefillRuntime.maximumAbsoluteDifference(
                    gpuOutput, allOutput
                )
            )
            gpuTopToken = Self.topToken(gpuOutput)
            allTopToken = Self.topToken(allOutput)
        }

        let gpuSummary = SurgicalPrefillRuntime.summary(gpuTimes)
        let allSummary = SurgicalPrefillRuntime.summary(allTimes)
        let monolithHash = artifacts.first {
            $0.name == "monolithic_layer_stack_512"
        }!.sha256
        let bootstrap = Self.pairedBootstrap(
            baseline: gpuTimes,
            candidate: allTimes,
            resamples: Self.bootstrapResamples,
            seed: Self.bootstrapSeed,
            signedDeltaDefinition: "M_GPU median - M_ALL median; positive favors M_ALL"
        )
        let allDispatch = dispatch.first { $0.name == "monolithic_all" }!
        let heterogeneous = Self.hasRegisteredHeterogeneousPattern(allDispatch)
        let noRegression = allSummary.median <= gpuSummary.median
            && bootstrap.upper95Ms >= 0.0
        let numerics = outputMaxAbs <= Self.maximumLogitError
            && gpuTopToken == allTopToken
        let passedG0a = heterogeneous && noRegression && numerics
        let thermalAfter = Self.thermalStateName(ProcessInfo.processInfo.thermalState)
        guard thermalAfter != "serious", thermalAfter != "critical" else {
            throw RuntimeError.invalidInput(
                "discarding Phase 0 timing after \(thermalAfter) thermal state"
            )
        }

        let monolithComparison = SelectiveMonolithComparison(
                mGPU: SelectiveCandidateTiming(
                    name: "M_GPU",
                    computeUnits: "CPU_AND_GPU",
                    packageSha256: monolithHash,
                    timesMs: gpuTimes,
                    medianMs: gpuSummary.median,
                    iqrMs: gpuSummary.iqr
                ),
                mALL: SelectiveCandidateTiming(
                    name: "M_ALL",
                    computeUnits: "ALL",
                    packageSha256: monolithHash,
                    timesMs: allTimes,
                    medianMs: allSummary.median,
                    iqrMs: allSummary.iqr
                ),
                runOrder: runOrder,
                improvement: bootstrap,
                outputMaxAbs: outputMaxAbs,
                mGPUTopToken: gpuTopToken,
                mALLTopToken: allTopToken,
                heterogeneousDispatch: heterogeneous,
                noSupportedRegression: noRegression,
                numericsPassed: numerics,
                passedG0a: passedG0a
            )
        if passedG0a {
            return makeResult(
                artifacts: artifacts,
                dispatch: dispatch,
                warmups: warmups,
                runs: runs,
                thermalBefore: thermalBefore,
                hostNoiseBefore: hostNoiseBefore,
                thermalAfter: thermalAfter,
                monolith: monolithComparison,
                pairs: [],
                aggregate: nil,
                gates: SelectiveG0Outcomes(
                    g0a: true, g0b: nil, g0c: nil, g0d: nil
                ),
                terminalGate: "G0a",
                verdict: "NO SPLIT NEEDED",
                continuePhase0: false,
                cancelled: [
                    "conv-pair GPU-versus-ANE timing",
                    "selective tail export",
                    "six-piece Mac benchmark",
                    "all physical-device latency phases",
                ]
            )
        }

        let fixedArtifacts = try fixedPairArtifactIdentities()
        let pairDispatch = try loadAndValidatePairDispatch(
            artifacts: fixedArtifacts
        )
        let g0b = Self.pairAdmissionPassed(pairDispatch)
        guard g0b else {
            return makeResult(
                artifacts: artifacts + fixedArtifacts,
                dispatch: dispatch + pairDispatch,
                warmups: warmups,
                runs: runs,
                thermalBefore: thermalBefore,
                hostNoiseBefore: hostNoiseBefore,
                thermalAfter: thermalAfter,
                monolith: monolithComparison,
                pairs: [],
                aggregate: nil,
                gates: SelectiveG0Outcomes(
                    g0a: false, g0b: false, g0c: nil, g0d: nil
                ),
                terminalGate: "G0b",
                verdict: "KILL",
                continuePhase0: false,
                cancelled: [
                    "conv-pair GPU-versus-ANE timing",
                    "selective tail export",
                    "all later phases",
                ]
            )
        }

        let oracleArtifacts = try oracleArtifactIdentities()
        let oracle = try loadOracleSegments()
        let entries = try realPairEntries(oracle: oracle)
        let validatedPairs = try benchmarkPairs(
            definitions: Self.pairDefinitions,
            entries: entries,
            oracle: oracle,
            warmups: warmups,
            runs: runs,
            measure: false
        )
        let g0d = validatedPairs.allSatisfy(\.numericsPassed)
        guard g0d else {
            return makeResult(
                artifacts: artifacts + fixedArtifacts + oracleArtifacts,
                dispatch: dispatch + pairDispatch,
                warmups: warmups,
                runs: runs,
                thermalBefore: thermalBefore,
                hostNoiseBefore: hostNoiseBefore,
                thermalAfter: Self.thermalStateName(ProcessInfo.processInfo.thermalState),
                monolith: monolithComparison,
                pairs: validatedPairs,
                aggregate: nil,
                gates: SelectiveG0Outcomes(
                    g0a: false, g0b: true, g0c: nil, g0d: false
                ),
                terminalGate: "G0d",
                verdict: "KILL",
                continuePhase0: false,
                cancelled: [
                    "conv-pair GPU-versus-ANE timing",
                    "aggregate pair economics",
                    "selective tail export",
                    "all later phases",
                ]
            )
        }
        let pairResults = try benchmarkPairs(
            definitions: Self.pairDefinitions,
            entries: entries,
            oracle: oracle,
            warmups: warmups,
            runs: runs,
            measure: true
        )
        let aggregate = Self.aggregatePairedBootstrap(
            pairResults.map { ($0.gpu.timesMs, $0.ane.timesMs) },
            resamples: Self.bootstrapResamples,
            seed: Self.bootstrapSeed ^ 0xA66E_6A7E
        )
        let g0c = aggregate.lower95Ms > 0.0
        let terminalGate: String? = g0c ? nil : "G0c"
        let verdict = g0c ? "GO" : "KILL"
        let finalThermalState = Self.thermalStateName(ProcessInfo.processInfo.thermalState)
        guard finalThermalState != "serious", finalThermalState != "critical" else {
            throw RuntimeError.invalidInput(
                "discarding pair timing after \(finalThermalState) thermal state"
            )
        }
        return makeResult(
            artifacts: artifacts + fixedArtifacts + oracleArtifacts,
            dispatch: dispatch + pairDispatch,
            warmups: warmups,
            runs: runs,
            thermalBefore: thermalBefore,
            hostNoiseBefore: hostNoiseBefore,
            thermalAfter: finalThermalState,
            monolith: monolithComparison,
            pairs: pairResults,
            aggregate: aggregate,
            gates: SelectiveG0Outcomes(
                g0a: false, g0b: true, g0c: g0c, g0d: true
            ),
            terminalGate: terminalGate,
            verdict: verdict,
            continuePhase0: verdict == "GO",
            cancelled: verdict == "KILL"
                ? ["selective tail export", "all later phases"]
                : []
        )
    }

    /// Builds one complete Phase 0 payload without duplicating evidence fields.
    private func makeResult(
        artifacts: [ArtifactIdentity],
        dispatch: [SelectiveDispatchEvidence],
        warmups: Int,
        runs: Int,
        thermalBefore: String,
        hostNoiseBefore: HostNoiseEvidence,
        thermalAfter: String,
        monolith: SelectiveMonolithComparison,
        pairs: [SelectivePairComparison],
        aggregate: PairedBootstrapEvidence?,
        gates: SelectiveG0Outcomes,
        terminalGate: String?,
        verdict: String,
        continuePhase0: Bool,
        cancelled: [String]
    ) -> SelectivePhase0Result {
        SelectivePhase0Result(
            schemaVersion: 2,
            runtime: "Release Swift CoreML direct fp16 MLMultiArray",
            checkpoint: input.checkpoint,
            environment: RuntimeEnvironment(
                host: ProcessInfo.processInfo.hostName,
                operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
                coremltools: input.coremltools
            ),
            artifacts: artifacts,
            dispatch: dispatch,
            bucket: input.bucket,
            prompt: input.prompt,
            warmupsPerCandidate: warmups,
            measuredRunsPerCandidate: runs,
            thermalStateBefore: thermalBefore,
            hostNoiseBefore: hostNoiseBefore,
            thermalStateAfter: thermalAfter,
            monolith: monolith,
            pairs: pairs,
            aggregatePairSavings: aggregate,
            gates: gates,
            terminalGate: terminalGate,
            verdict: verdict,
            continuePhase0: continuePhase0,
            cancelledMeasurements: cancelled
        )
    }

    /// Runs embedding, the supplied layer stack, and the common head.
    private func runMonolith(_ model: MLModel) throws -> MLMultiArray {
        let hidden = try SurgicalPrefillRuntime.predict(
            model: embedding,
            values: ["token_ids": tokenIDs, "token_mask": tokenMask],
            output: "hidden_out"
        )
        try SurgicalPrefillRuntime.requireFloat16Boundary(
            hidden, label: "embedding hidden output"
        )
        var values: [String: MLMultiArray] = [
            "hidden_states": hidden,
            "cosine": cosine,
            "sine": sine,
            "attention_mask": attentionMask,
        ]
        for layer in [0, 1, 3, 4, 6, 7, 9, 11, 13, 15] {
            values["conv_state_\(layer)_in"] = zeroState
        }
        let output = try SurgicalPrefillRuntime.predict(
            model: model, values: values, output: "hidden_out"
        )
        try SurgicalPrefillRuntime.requireFloat16Boundary(
            output, label: "monolithic hidden output"
        )
        return try SurgicalPrefillRuntime.runHead(
            model: languageModelHead,
            hidden: output,
            lastPosition: input.bucket - 1
        )
    }

    /// Loads the frozen 13-segment GPU chain only after G0a and G0b continue.
    private func loadOracleSegments() throws -> [(SegmentDescriptor, MLModel)] {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU
        return try SurgicalPrefillRuntime.segments.map { descriptor in
            let package = stage1Directory
                .appendingPathComponent("segments", isDirectory: true)
                .appendingPathComponent("\(descriptor.name).mlpackage")
            return (
                descriptor,
                try SurgicalPrefillRuntime.loadModel(
                    package: package, configuration: configuration
                )
            )
        }
    }

    /// Produces each pair's real layer-entry tensor with the untimed GPU oracle.
    private func realPairEntries(
        oracle: [(SegmentDescriptor, MLModel)]
    ) throws -> [String: MLMultiArray] {
        var hidden = try SurgicalPrefillRuntime.predict(
            model: embedding,
            values: ["token_ids": tokenIDs, "token_mask": tokenMask],
            output: "hidden_out"
        )
        var entries: [String: MLMultiArray] = [:]
        for segmentIndex in 0...4 {
            if let pair = Self.pairDefinitions.first(
                where: { $0.segmentIndex == segmentIndex }
            ) {
                entries[pair.name] = hidden
            }
            if segmentIndex < 4 {
                hidden = try runSegment(
                    descriptor: oracle[segmentIndex].0,
                    model: oracle[segmentIndex].1,
                    hidden: hidden
                )
            }
        }
        guard entries.count == Self.pairDefinitions.count else {
            throw RuntimeError.invalidInput("failed to produce all real pair entries")
        }
        return entries
    }

    /// Runs one frozen segment with the shared attention/state tensor contract.
    private func runSegment(
        descriptor: SegmentDescriptor,
        model: MLModel,
        hidden: MLMultiArray
    ) throws -> MLMultiArray {
        var values: [String: MLMultiArray] = ["hidden_states": hidden]
        if descriptor.usesAttentionInputs {
            values["cosine"] = cosine
            values["sine"] = sine
            values["attention_mask"] = attentionMask
        } else {
            for name in descriptor.stateInputs {
                values[name] = zeroState
            }
        }
        let output = try SurgicalPrefillRuntime.predict(
            model: model, values: values, output: "hidden_out"
        )
        try SurgicalPrefillRuntime.requireFloat16Boundary(
            output, label: "\(descriptor.name) hidden output"
        )
        return output
    }

    /// Benchmarks all three fixed twins after source and GPU/ANE parity checks.
    private func benchmarkPairs(
        definitions: [PairDefinition],
        entries: [String: MLMultiArray],
        oracle: [(SegmentDescriptor, MLModel)],
        warmups: Int,
        runs: Int,
        measure: Bool
    ) throws -> [SelectivePairComparison] {
        var results: [SelectivePairComparison] = []
        for (pairOffset, definition) in definitions.enumerated() {
            let package = fixedPairDirectory.appendingPathComponent(
                "\(definition.name).mlpackage"
            )
            let packageHash = try SurgicalPrefillRuntime.packageTreeSHA256(package)
            let sourcePackage = stage1Directory
                .appendingPathComponent("segments", isDirectory: true)
                .appendingPathComponent("\(definition.name).mlpackage")
            let sourceHash = try SurgicalPrefillRuntime.packageTreeSHA256(sourcePackage)
            let gpuConfiguration = MLModelConfiguration()
            gpuConfiguration.computeUnits = .cpuAndGPU
            let aneConfiguration = MLModelConfiguration()
            aneConfiguration.computeUnits = .cpuAndNeuralEngine
            let validationModel = try SurgicalPrefillRuntime.loadModel(
                package: package, configuration: gpuConfiguration
            )
            let entry = entries[definition.name]!
            let sourceOutputs = try pairOutputs(
                model: oracle[definition.segmentIndex].1,
                definition: definition,
                hidden: entry
            )
            let twinOutputs = try pairOutputs(
                model: validationModel,
                definition: definition,
                hidden: entry
            )
            let sourceTwinMaxAbs = try Self.maximumOutputDifference(
                sourceOutputs, twinOutputs
            )
            guard sourceTwinMaxAbs <= Self.maximumLogitError else {
                throw RuntimeError.invalidInput(
                    "\(definition.name) materialization parity failed: \(sourceTwinMaxAbs)"
                )
            }

            let gpuModel = try SurgicalPrefillRuntime.loadModel(
                package: package, configuration: gpuConfiguration
            )
            let aneModel = try SurgicalPrefillRuntime.loadModel(
                package: package, configuration: aneConfiguration
            )
            let gpuCheck = try pairOutputs(
                model: gpuModel, definition: definition, hidden: entry
            )
            let aneCheck = try pairOutputs(
                model: aneModel, definition: definition, hidden: entry
            )
            let outputMaxAbs = try Self.maximumOutputDifference(gpuCheck, aneCheck)
            let gpuLogits = try suffixLogits(
                hidden: gpuCheck["hidden_out"]!,
                after: definition.segmentIndex,
                oracle: oracle
            )
            let aneLogits = try suffixLogits(
                hidden: aneCheck["hidden_out"]!,
                after: definition.segmentIndex,
                oracle: oracle
            )
            let finalLogitMaxAbs = try SurgicalPrefillRuntime.maximumAbsoluteDifference(
                gpuLogits, aneLogits
            )
            let gpuTopToken = Self.topToken(gpuLogits)
            let aneTopToken = Self.topToken(aneLogits)
            let numericsPassed = sourceTwinMaxAbs <= Self.maximumLogitError
                && outputMaxAbs <= Self.maximumLogitError
                && finalLogitMaxAbs <= Self.maximumLogitError
                && gpuTopToken == aneTopToken
            guard measure else {
                results.append(
                    SelectivePairComparison(
                        name: definition.name,
                        sourcePackageSha256: sourceHash,
                        sourceTwinMaxAbs: sourceTwinMaxAbs,
                        gpu: SelectiveCandidateTiming(
                            name: "\(definition.name)_GPU",
                            computeUnits: "CPU_AND_GPU",
                            packageSha256: packageHash,
                            timesMs: [],
                            medianMs: nil,
                            iqrMs: nil
                        ),
                        ane: SelectiveCandidateTiming(
                            name: "\(definition.name)_ANE",
                            computeUnits: "CPU_AND_NE",
                            packageSha256: packageHash,
                            timesMs: [],
                            medianMs: nil,
                            iqrMs: nil
                        ),
                        runOrder: [],
                        improvement: nil,
                        outputMaxAbs: outputMaxAbs,
                        finalLogitMaxAbs: finalLogitMaxAbs,
                        gpuTopToken: gpuTopToken,
                        aneTopToken: aneTopToken,
                        numericsPassed: numericsPassed
                    )
                )
                continue
            }
            guard numericsPassed else {
                throw RuntimeError.invalidInput(
                    "\(definition.name) numerics changed after preflight validation"
                )
            }

            for _ in 0..<warmups {
                _ = try pairOutputs(
                    model: gpuModel, definition: definition, hidden: entry
                )
                _ = try pairOutputs(
                    model: aneModel, definition: definition, hidden: entry
                )
            }
            var gpuTimes: [Double] = []
            var aneTimes: [Double] = []
            var runOrder: [String] = []
            for runIndex in 0..<runs {
                let order = runIndex.isMultiple(of: 2)
                    ? ["GPU", "ANE"]
                    : ["ANE", "GPU"]
                runOrder.append(order.joined(separator: "->"))
                for candidate in order {
                    let model = candidate == "GPU" ? gpuModel : aneModel
                    let start = DispatchTime.now().uptimeNanoseconds
                    _ = try pairOutputs(
                        model: model, definition: definition, hidden: entry
                    )
                    let elapsed = Double(
                        DispatchTime.now().uptimeNanoseconds - start
                    ) / 1_000_000.0
                    if candidate == "GPU" {
                        gpuTimes.append(elapsed)
                    } else {
                        aneTimes.append(elapsed)
                    }
                }
            }
            let gpuSummary = SurgicalPrefillRuntime.summary(gpuTimes)
            let aneSummary = SurgicalPrefillRuntime.summary(aneTimes)
            let improvement = Self.pairedBootstrap(
                baseline: gpuTimes,
                candidate: aneTimes,
                resamples: Self.bootstrapResamples,
                seed: Self.bootstrapSeed &+ UInt64(pairOffset + 1),
                signedDeltaDefinition:
                    "\(definition.name) GPU median - ANE median; positive favors ANE"
            )
            results.append(
                SelectivePairComparison(
                    name: definition.name,
                    sourcePackageSha256: sourceHash,
                    sourceTwinMaxAbs: sourceTwinMaxAbs,
                    gpu: SelectiveCandidateTiming(
                        name: "\(definition.name)_GPU",
                        computeUnits: "CPU_AND_GPU",
                        packageSha256: packageHash,
                        timesMs: gpuTimes,
                        medianMs: gpuSummary.median,
                        iqrMs: gpuSummary.iqr
                    ),
                    ane: SelectiveCandidateTiming(
                        name: "\(definition.name)_ANE",
                        computeUnits: "CPU_AND_NE",
                        packageSha256: packageHash,
                        timesMs: aneTimes,
                        medianMs: aneSummary.median,
                        iqrMs: aneSummary.iqr
                    ),
                    runOrder: runOrder,
                    improvement: improvement,
                    outputMaxAbs: outputMaxAbs,
                    finalLogitMaxAbs: finalLogitMaxAbs,
                    gpuTopToken: gpuTopToken,
                    aneTopToken: aneTopToken,
                    numericsPassed: numericsPassed
                )
            )
        }
        return results
    }

    /// Returns independent snapshots of every documented pair output.
    private func pairOutputs(
        model: MLModel,
        definition: PairDefinition,
        hidden: MLMultiArray
    ) throws -> [String: MLMultiArray] {
        var values: [String: MLMultiArray] = ["hidden_states": hidden]
        for name in definition.stateInputs {
            values[name] = zeroState
        }
        let prediction = try SurgicalPrefillRuntime.predictAll(
            model: model, values: values
        )
        return try Dictionary(
            uniqueKeysWithValues: definition.outputNames.map { name in
                guard let output = prediction.featureValue(for: name)?.multiArrayValue else {
                    throw RuntimeError.missingOutput(name)
                }
                if name == "hidden_out" {
                    try SurgicalPrefillRuntime.requireFloat16Boundary(
                        output, label: "\(definition.name) hidden output"
                    )
                }
                return (name, try SurgicalPrefillRuntime.copyMultiArray(output))
            }
        )
    }

    /// Runs the common untimed GPU suffix to obtain a meaningful top token.
    private func suffixLogits(
        hidden initialHidden: MLMultiArray,
        after segmentIndex: Int,
        oracle: [(SegmentDescriptor, MLModel)]
    ) throws -> MLMultiArray {
        var hidden = initialHidden
        if segmentIndex + 1 < oracle.count {
            for index in (segmentIndex + 1)..<oracle.count {
                hidden = try runSegment(
                    descriptor: oracle[index].0,
                    model: oracle[index].1,
                    hidden: hidden
                )
            }
        }
        return try SurgicalPrefillRuntime.runHead(
            model: languageModelHead,
            hidden: hidden,
            lastPosition: input.bucket - 1
        )
    }

    /// Returns unique package identities shared across both policies.
    private func artifactIdentities() throws -> [ArtifactIdentity] {
        try [
            "embedding.mlpackage",
            "lm_head.mlpackage",
            "monolithic_layer_stack_512.mlpackage",
        ].map { path in
            let package = stage1Directory.appendingPathComponent(path)
            return ArtifactIdentity(
                name: package.deletingPathExtension().lastPathComponent,
                path: path,
                sha256: try SurgicalPrefillRuntime.packageTreeSHA256(package)
            )
        }
    }

    /// Hashes the three exact fixed twins used by both pair policies.
    private func fixedPairArtifactIdentities() throws -> [ArtifactIdentity] {
        try Self.pairDefinitions.map { definition in
            let package = fixedPairDirectory.appendingPathComponent(
                "\(definition.name).mlpackage"
            )
            return ArtifactIdentity(
                name: "\(definition.name)_fixed512",
                path: "fixed_pairs_512/\(definition.name).mlpackage",
                sha256: try SurgicalPrefillRuntime.packageTreeSHA256(package)
            )
        }
    }

    /// Hashes every Stage 1 segment used for real entries and suffix logits.
    private func oracleArtifactIdentities() throws -> [ArtifactIdentity] {
        try SurgicalPrefillRuntime.segments.map { descriptor in
            let path = "segments/\(descriptor.name).mlpackage"
            let package = stage1Directory.appendingPathComponent(path)
            return ArtifactIdentity(
                name: descriptor.name,
                path: path,
                sha256: try SurgicalPrefillRuntime.packageTreeSHA256(package)
            )
        }
    }

    /// Loads the four G0a plans and rejects any host, OS, policy, or hash drift.
    private func loadAndValidateDispatch(
        artifacts: [ArtifactIdentity]
    ) throws -> [SelectiveDispatchEvidence] {
        let artifactHashes = Dictionary(
            uniqueKeysWithValues: artifacts.map { ($0.name, $0.sha256) }
        )
        let expected: [(String, String, String)] = [
            ("embedding_gpu", "embedding", "CPU_AND_GPU"),
            ("lm_head_gpu", "lm_head", "CPU_AND_GPU"),
            ("monolithic_gpu", "monolithic_layer_stack_512", "CPU_AND_GPU"),
            ("monolithic_all", "monolithic_layer_stack_512", "ALL"),
        ]
        return try expected.map { name, artifactName, units in
            let url = dispatchDirectory.appendingPathComponent("\(name).json")
            let data = try Data(contentsOf: url)
            let document = try JSONDecoder().decode(ComputePlanDocument.self, from: data)
            guard document.provenance.whereRun == "host" else {
                throw RuntimeError.invalidInput("\(name) is not a host compute plan")
            }
            guard document.provenance.host.lowercased()
                    == ProcessInfo.processInfo.hostName.lowercased() else {
                throw RuntimeError.invalidInput("\(name) host does not match timed process")
            }
            guard document.provenance.macosBuild == Self.currentOSBuild else {
                throw RuntimeError.invalidInput("\(name) OS build does not match timed process")
            }
            guard document.provenance.coremltools == input.coremltools else {
                throw RuntimeError.invalidInput("\(name) coremltools version drifted")
            }
            guard document.provenance.computeUnits == units else {
                throw RuntimeError.invalidInput("\(name) compute units drifted")
            }
            guard let expectedHash = artifactHashes[artifactName],
                  document.provenance.packageSha256 == expectedHash else {
                throw RuntimeError.invalidInput("\(name) package hash drifted")
            }
            return SelectiveDispatchEvidence(
                name: name,
                path: "phase0_dispatch/\(name).json",
                evidenceSha256: SHA256.hash(data: data).map {
                    String(format: "%02x", $0)
                }.joined(),
                packageSha256: document.provenance.packageSha256,
                computeUnits: units,
                totalCostedOps: document.totalCostedOps,
                preferredDeviceHistogram: document.preferredDeviceHistogram,
                perOpType: document.perOpType
            )
        }
    }

    /// Loads six fixed-512 pair plans and validates exact package provenance.
    private func loadAndValidatePairDispatch(
        artifacts: [ArtifactIdentity]
    ) throws -> [SelectiveDispatchEvidence] {
        var evidence: [SelectiveDispatchEvidence] = []
        for definition in Self.pairDefinitions {
            let artifactName = "\(definition.name)_fixed512"
            guard let artifact = artifacts.first(where: { $0.name == artifactName }) else {
                throw RuntimeError.invalidInput("missing \(artifactName) identity")
            }
            for (suffix, units) in [
                ("gpu", "CPU_AND_GPU"),
                ("ane", "CPU_AND_NE"),
            ] {
                let name = "\(definition.name)_fixed512_\(suffix)"
                let url = dispatchDirectory.appendingPathComponent("\(name).json")
                let data = try Data(contentsOf: url)
                let document = try JSONDecoder().decode(
                    ComputePlanDocument.self, from: data
                )
                try validatePlanProvenance(
                    document.provenance,
                    name: name,
                    expectedUnits: units,
                    expectedPackageHash: artifact.sha256
                )
                evidence.append(
                    SelectiveDispatchEvidence(
                        name: name,
                        path: "phase0_dispatch/\(name).json",
                        evidenceSha256: SHA256.hash(data: data).map {
                            String(format: "%02x", $0)
                        }.joined(),
                        packageSha256: artifact.sha256,
                        computeUnits: units,
                        totalCostedOps: document.totalCostedOps,
                        preferredDeviceHistogram: document.preferredDeviceHistogram,
                        perOpType: document.perOpType
                    )
                )
            }
        }
        return evidence
    }

    /// Applies the same fail-closed host, toolchain, policy, and hash checks.
    private func validatePlanProvenance(
        _ provenance: ComputePlanProvenance,
        name: String,
        expectedUnits: String,
        expectedPackageHash: String
    ) throws {
        guard provenance.whereRun == "host" else {
            throw RuntimeError.invalidInput("\(name) is not a host compute plan")
        }
        guard provenance.host.lowercased()
                == ProcessInfo.processInfo.hostName.lowercased() else {
            throw RuntimeError.invalidInput("\(name) host does not match timed process")
        }
        guard provenance.macosBuild == Self.currentOSBuild else {
            throw RuntimeError.invalidInput("\(name) OS build does not match timed process")
        }
        guard provenance.coremltools == input.coremltools else {
            throw RuntimeError.invalidInput("\(name) coremltools version drifted")
        }
        guard provenance.computeUnits == expectedUnits else {
            throw RuntimeError.invalidInput("\(name) compute units drifted")
        }
        guard provenance.packageSha256 == expectedPackageHash else {
            throw RuntimeError.invalidInput("\(name) package hash drifted")
        }
    }

    /// Requires every ANE-permitted fixed pair to exceed the frozen 80% gate.
    private static func pairAdmissionPassed(
        _ evidence: [SelectiveDispatchEvidence]
    ) -> Bool {
        let anePlans = evidence.filter { $0.computeUnits == "CPU_AND_NE" }
        return anePlans.count == pairDefinitions.count && anePlans.allSatisfy {
            guard $0.totalCostedOps > 0 else { return false }
            return Double($0.preferredDeviceHistogram["neuralEngine"] ?? 0)
                / Double($0.totalCostedOps) >= 0.80
        }
    }

    /// Requires a broad conv-on-ANE pattern while attention-defining ops stay GPU.
    private static func hasRegisteredHeterogeneousPattern(
        _ evidence: SelectiveDispatchEvidence
    ) -> Bool {
        let conv = evidence.perOpType.first { $0.key.hasSuffix(".conv") }?.value ?? [:]
        let matmul = evidence.perOpType.first {
            $0.key.hasSuffix(".matmul")
        }?.value ?? [:]
        let softmax = evidence.perOpType.first {
            $0.key.hasSuffix(".softmax")
        }?.value ?? [:]
        return (conv["neuralEngine"] ?? 0) > 0
            && (matmul["gpu"] ?? 0) > 0
            && (matmul["neuralEngine"] ?? 0) == 0
            && (softmax["gpu"] ?? 0) > 0
            && (softmax["neuralEngine"] ?? 0) == 0
    }

    /// Returns the argmax index without materializing an intermediate array.
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

    /// Paired percentile bootstrap of the registered difference-of-medians.
    private static func pairedBootstrap(
        baseline: [Double],
        candidate: [Double],
        resamples: Int,
        seed: UInt64,
        signedDeltaDefinition: String
    ) -> PairedBootstrapEvidence {
        precondition(baseline.count == candidate.count && !baseline.isEmpty)
        let baselineMedian = SurgicalPrefillRuntime.summary(baseline).median
        let candidateMedian = SurgicalPrefillRuntime.summary(candidate).median
        var generator = SplitMix64(state: seed)
        var estimates: [Double] = []
        estimates.reserveCapacity(resamples)
        for _ in 0..<resamples {
            var baselineSample: [Double] = []
            var candidateSample: [Double] = []
            baselineSample.reserveCapacity(baseline.count)
            candidateSample.reserveCapacity(candidate.count)
            for _ in baseline.indices {
                let index = generator.nextIndex(upperBound: baseline.count)
                baselineSample.append(baseline[index])
                candidateSample.append(candidate[index])
            }
            estimates.append(
                SurgicalPrefillRuntime.summary(baselineSample).median
                    - SurgicalPrefillRuntime.summary(candidateSample).median
            )
        }
        estimates.sort()
        return PairedBootstrapEvidence(
            signedDeltaDefinition: signedDeltaDefinition,
            pairedDeltasMs: zip(baseline, candidate).map(-),
            pointEstimateMs: baselineMedian - candidateMedian,
            lower95Ms: SurgicalPrefillRuntime.percentile(estimates, fraction: 0.025),
            upper95Ms: SurgicalPrefillRuntime.percentile(estimates, fraction: 0.975),
            resamples: resamples,
            seed: seed
        )
    }

    /// Computes the largest absolute difference across the exact named outputs.
    private static func maximumOutputDifference(
        _ baseline: [String: MLMultiArray],
        _ candidate: [String: MLMultiArray]
    ) throws -> Double {
        guard Set(baseline.keys) == Set(candidate.keys) else {
            throw RuntimeError.invalidInput("pair output names do not match")
        }
        var maximum = 0.0
        for name in baseline.keys.sorted() {
            maximum = max(
                maximum,
                try SurgicalPrefillRuntime.maximumAbsoluteDifference(
                    baseline[name]!, candidate[name]!
                )
            )
        }
        return maximum
    }

    /// Bootstraps the registered sum of the three pair median savings.
    private static func aggregatePairedBootstrap(
        _ pairs: [([Double], [Double])],
        resamples: Int,
        seed: UInt64
    ) -> PairedBootstrapEvidence {
        precondition(!pairs.isEmpty)
        precondition(pairs.allSatisfy { !$0.0.isEmpty && $0.0.count == $0.1.count })
        let pointEstimate = pairs.reduce(0.0) { partial, pair in
            partial + SurgicalPrefillRuntime.summary(pair.0).median
                - SurgicalPrefillRuntime.summary(pair.1).median
        }
        let pairedDeltas = pairs[0].0.indices.map { index in
            pairs.reduce(0.0) { partial, pair in
                partial + pair.0[index] - pair.1[index]
            }
        }
        var generator = SplitMix64(state: seed)
        var estimates: [Double] = []
        estimates.reserveCapacity(resamples)
        for _ in 0..<resamples {
            var aggregate = 0.0
            for pair in pairs {
                var gpuSample: [Double] = []
                var aneSample: [Double] = []
                gpuSample.reserveCapacity(pair.0.count)
                aneSample.reserveCapacity(pair.1.count)
                for _ in pair.0.indices {
                    let index = generator.nextIndex(upperBound: pair.0.count)
                    gpuSample.append(pair.0[index])
                    aneSample.append(pair.1[index])
                }
                aggregate += SurgicalPrefillRuntime.summary(gpuSample).median
                    - SurgicalPrefillRuntime.summary(aneSample).median
            }
            estimates.append(aggregate)
        }
        estimates.sort()
        return PairedBootstrapEvidence(
            signedDeltaDefinition:
                "sum(pair GPU medians - pair ANE medians); positive favors ANE",
            pairedDeltasMs: pairedDeltas,
            pointEstimateMs: pointEstimate,
            lower95Ms: SurgicalPrefillRuntime.percentile(estimates, fraction: 0.025),
            upper95Ms: SurgicalPrefillRuntime.percentile(estimates, fraction: 0.975),
            resamples: resamples,
            seed: seed
        )
    }

    /// Captures and evaluates the frozen low-noise Mac timing preflight.
    private static func captureHostNoiseEvidence() throws -> HostNoiseEvidence {
        var loads = [Double](repeating: 0.0, count: 3)
        guard getloadavg(&loads, Int32(loads.count)) == loads.count else {
            throw RuntimeError.invalidInput("could not read host load averages")
        }
        let logicalProcessors = ProcessInfo.processInfo.activeProcessorCount
        let normalizedLoad = loads[0] / Double(logicalProcessors)

        let memoryOutput = try commandOutput(
            executable: "/usr/bin/memory_pressure", arguments: ["-Q"]
        )
        guard let memoryLine = memoryOutput.split(separator: "\n").first(
            where: { $0.contains("System-wide memory free percentage:") }
        ),
        let memoryPercent = Int(
            memoryLine.split(separator: ":").last?
                .trimmingCharacters(in: .whitespaces)
                .replacingOccurrences(of: "%", with: "") ?? ""
        ) else {
            throw RuntimeError.invalidInput("could not parse host memory pressure")
        }

        let swapOutput = try commandOutput(
            executable: "/usr/sbin/sysctl", arguments: ["-n", "vm.swapusage"]
        )
        guard let usedRange = swapOutput.range(of: "used = ") else {
            throw RuntimeError.invalidInput("could not parse host swap usage")
        }
        let usedSuffix = swapOutput[usedRange.upperBound...]
        let usedText = usedSuffix.prefix { $0.isNumber || $0 == "." }
        guard let swapUsed = Double(usedText) else {
            throw RuntimeError.invalidInput("could not parse host swap amount")
        }

        let processOutput = try commandOutput(
            executable: "/bin/ps",
            arguments: ["-A", "-o", "pid=", "-o", "%cpu=", "-o", "comm="]
        )
        let currentPID = ProcessInfo.processInfo.processIdentifier
        let highCPU: [HostProcessNoise] = processOutput.split(separator: "\n").compactMap { line in
            let fields = line.split(
                maxSplits: 2, whereSeparator: { $0.isWhitespace }
            )
            guard fields.count == 3,
                  let pid = Int32(fields[0]),
                  let cpu = Double(fields[1]),
                  pid != currentPID,
                  cpu >= 50.0 else {
                return nil
            }
            return HostProcessNoise(
                pid: pid, cpuPercent: cpu, command: String(fields[2])
            )
        }
        let passed = normalizedLoad <= 0.5
            && memoryPercent >= 15
            && highCPU.isEmpty
        return HostNoiseEvidence(
            capturedAtUTC: ISO8601DateFormatter().string(from: Date()),
            logicalProcessorCount: logicalProcessors,
            loadAverages: loads,
            oneMinuteLoadPerProcessor: normalizedLoad,
            memoryFreePercent: memoryPercent,
            swapUsedMiB: swapUsed,
            highCPUProcesses: highCPU,
            policy: "load1/logicalCPU <= 0.5; free memory >= 15%; no other process >= 50% CPU",
            passed: passed
        )
    }

    /// Runs one read-only host probe and returns UTF-8 standard output.
    private static func commandOutput(
        executable: String, arguments: [String]
    ) throws -> String {
        #if os(macOS)
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw RuntimeError.invalidInput(
                "host probe failed: \(executable) \(arguments.joined(separator: " "))"
            )
        }
        return String(decoding: data, as: UTF8.self)
        #else
        throw RuntimeError.invalidInput(
            "Mac host-noise probes are unavailable on this platform"
        )
        #endif
    }

    /// Extracts the kernel build identifier from ProcessInfo's stable string.
    private static var currentOSBuild: String {
        let description = ProcessInfo.processInfo.operatingSystemVersionString
        guard let start = description.range(of: "Build ") else { return description }
        let suffix = description[start.upperBound...]
        return String(suffix.prefix { $0 != ")" })
    }

    /// Serializes ProcessInfo thermal state without private APIs.
    private static func thermalStateName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

/// Frozen metadata for one registered two-convolution island.
struct PairDefinition {
    /// Package basename and durable evidence label.
    let name: String

    /// Index of the equivalent segment in the 13-segment oracle.
    let segmentIndex: Int

    /// Two zero-state inputs owned by this convolution pair.
    let stateInputs: [String]

    /// Hidden and state outputs required for source/twin parity.
    let outputNames: [String]
}

/// Minimal decoder for the compute-plan evidence schema.
private struct ComputePlanDocument: Decodable {
    let totalCostedOps: Int
    let preferredDeviceHistogram: [String: Int]
    let perOpType: [String: [String: Int]]
    let provenance: ComputePlanProvenance

    private enum CodingKeys: String, CodingKey {
        case totalCostedOps = "total_costed_ops"
        case preferredDeviceHistogram = "preferred_device_histogram"
        case perOpType = "per_op_type"
        case provenance
    }
}

/// Provenance fields required to bind a plan to the timed process.
private struct ComputePlanProvenance: Decodable {
    let whereRun: String
    let host: String
    let macosBuild: String
    let packageSha256: String
    let computeUnits: String
    let coremltools: String

    private enum CodingKeys: String, CodingKey {
        case whereRun = "where"
        case host
        case macosBuild = "macos_build"
        case packageSha256 = "package_sha256"
        case computeUnits = "compute_units"
        case coremltools
    }
}

/// Small deterministic generator used only for reproducible resampling.
private struct SplitMix64 {
    var state: UInt64

    mutating func nextIndex(upperBound: Int) -> Int {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        value ^= value >> 31
        return Int(value % UInt64(upperBound))
    }
}
