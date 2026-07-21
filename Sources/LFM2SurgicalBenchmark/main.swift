import Foundation
import LFM2SurgicalRuntime

/// Minimal command-line options for the frozen G1a benchmark.
private struct Options {
    /// Directory containing generated Stage 1 packages.
    let stage1Directory: URL

    /// JSON fixture written by `prepare_stage1_input.py`.
    let inputURL: URL

    /// Number of untimed predictions per candidate.
    let warmups: Int

    /// Number of counterbalanced measured predictions per candidate.
    let runs: Int

    /// Durable JSON evidence destination.
    let outputURL: URL

    /// Parses explicit named arguments and rejects missing values.
    static func parse(_ arguments: [String]) throws -> Options {
        var values: [String: String] = [:]
        var index = 0
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
            let output = values["--out"],
            let warmupsText = values["--warmups"],
            let runsText = values["--runs"],
            let warmups = Int(warmupsText),
            let runs = Int(runsText)
        else {
            throw RuntimeError.invalidInput(
                "required: --stage1-dir PATH --input-json PATH --warmups N --runs N --out PATH"
            )
        }
        let workingDirectory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return Options(
            stage1Directory: URL(fileURLWithPath: stage1, relativeTo: workingDirectory).standardizedFileURL,
            inputURL: URL(fileURLWithPath: input, relativeTo: workingDirectory).standardizedFileURL,
            warmups: warmups,
            runs: runs,
            outputURL: URL(fileURLWithPath: output, relativeTo: workingDirectory).standardizedFileURL
        )
    }
}

do {
    let options = try Options.parse(Array(CommandLine.arguments.dropFirst()))
    let runtime = try SurgicalPrefillRuntime(
        stage1Directory: options.stage1Directory,
        inputURL: options.inputURL
    )
    let result = try runtime.benchmark(warmups: options.warmups, runs: options.runs)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(result)
    try FileManager.default.createDirectory(
        at: options.outputURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try data.write(to: options.outputURL, options: .atomic)
    print(String(decoding: data, as: UTF8.self))
    if !result.passed {
        fputs("G1a failed: Swift segment overhead exceeds 30% of total prefill\n", stderr)
        exit(EXIT_FAILURE)
    }
} catch {
    fputs("lfm2-surgical-benchmark: \(error)\n", stderr)
    exit(EXIT_FAILURE)
}
