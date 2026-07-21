import Foundation
import LFM2SurgicalRuntime

/// Named arguments for the additive plan-011 benchmark.
private struct Options {
    let stage1Directory: URL
    let inputURL: URL
    let dispatchDirectory: URL
    let fixedPairDirectory: URL
    let warmups: Int
    let runs: Int
    let outputURL: URL

    /// Parses `phase0` and rejects every unregistered mode or missing value.
    static func parse(_ arguments: [String]) throws -> Options {
        guard arguments.first == "phase0" else {
            throw RuntimeError.invalidInput("required subcommand: phase0")
        }
        var values: [String: String] = [:]
        var index = 1
        while index < arguments.count {
            let key = arguments[index]
            guard key.hasPrefix("--"), index + 1 < arguments.count else {
                throw RuntimeError.invalidInput("expected --name value arguments")
            }
            values[key] = arguments[index + 1]
            index += 2
        }
        guard
            let stage1 = values["--stage1-dir"],
            let input = values["--input-json"],
            let dispatch = values["--dispatch-dir"],
            let fixedPairs = values["--fixed-pair-dir"],
            let output = values["--out"],
            let warmupsText = values["--warmups"],
            let runsText = values["--runs"],
            let warmups = Int(warmupsText),
            let runs = Int(runsText)
        else {
            throw RuntimeError.invalidInput(
                "required: phase0 --stage1-dir PATH --input-json PATH "
                    + "--dispatch-dir PATH --fixed-pair-dir PATH "
                    + "--warmups 3 --runs 20 --out PATH"
            )
        }
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return Options(
            stage1Directory: URL(fileURLWithPath: stage1, relativeTo: cwd).standardizedFileURL,
            inputURL: URL(fileURLWithPath: input, relativeTo: cwd).standardizedFileURL,
            dispatchDirectory: URL(fileURLWithPath: dispatch, relativeTo: cwd).standardizedFileURL,
            fixedPairDirectory: URL(
                fileURLWithPath: fixedPairs, relativeTo: cwd
            ).standardizedFileURL,
            warmups: warmups,
            runs: runs,
            outputURL: URL(fileURLWithPath: output, relativeTo: cwd).standardizedFileURL
        )
    }
}

do {
    let options = try Options.parse(Array(CommandLine.arguments.dropFirst()))
    let runtime = try SelectivePrefillRuntime(
        stage1Directory: options.stage1Directory,
        inputURL: options.inputURL,
        dispatchDirectory: options.dispatchDirectory,
        fixedPairDirectory: options.fixedPairDirectory
    )
    let result = try runtime.benchmarkPhase0(
        warmups: options.warmups,
        runs: options.runs
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(result)
    try FileManager.default.createDirectory(
        at: options.outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try data.write(to: options.outputURL, options: .atomic)
    print(String(decoding: data, as: UTF8.self))
} catch {
    fputs("lfm2-selective-benchmark: \(error)\n", stderr)
    exit(EXIT_FAILURE)
}
