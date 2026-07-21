import Foundation
import LFM2SurgicalRuntime
import SwiftUI

/// Versioned setup/runtime failure retrieved by the host as `INCONCLUSIVE`.
private struct A17ParityAppFailure: Codable {
    /// Failure schema version.
    let schemaVersion: Int

    /// Non-scientific terminal status for setup or runtime failure.
    let status: String

    /// Pair requested by the signed build manifest.
    let pair: String

    /// Actionable error text without credentials or raw signing material.
    let error: String
}

/// Minimal foreground surface; the experiment runs once from the launch task.
private struct A17ParityView: View {
    /// Human-readable status for an unlocked foreground device.
    @State private var status = "Preparing registered parity probe"

    /// The deliberately tiny app surface contains no controls or timing UI.
    var body: some View {
        VStack(spacing: 12) {
            Text("LFM2 A17 Parity")
                .font(.headline)
            Text(status)
                .multilineTextAlignment(.center)
                .font(.footnote)
        }
        .padding()
        .task {
            await runOnce()
        }
    }

    /// Executes once, then atomically persists either evidence or setup failure.
    @MainActor
    private func runOnce() async {
        do {
            status = "Running untimed correctness comparisons"
            let (result, filename) = try await Task.detached(priority: .userInitiated) {
                let runtime = try A17ParityRuntime()
                return (try runtime.run(), runtime.resultFilename)
            }.value
            try Self.write(result, filename: filename)
            status = result.verdict
        } catch {
            let manifest = try? Self.loadManifest()
            let filename = manifest?.resultFilename ?? "a17-parity-error.json"
            let failure = A17ParityAppFailure(
                schemaVersion: 1,
                status: "INCONCLUSIVE",
                pair: manifest?.pair ?? "unknown",
                error: String(describing: error)
            )
            try? Self.write(failure, filename: filename)
            status = "INCONCLUSIVE: \(error)"
        }
    }

    /// Decodes the signed build manifest for fail-closed error persistence.
    private static func loadManifest() throws -> A17ParityManifest {
        guard let url = Bundle.main.url(
            forResource: "a17-parity-manifest", withExtension: "json"
        ) else {
            throw CocoaError(.fileNoSuchFile)
        }
        return try JSONDecoder().decode(
            A17ParityManifest.self, from: Data(contentsOf: url)
        )
    }

    /// Writes stable snake-case JSON atomically under the app Documents domain.
    private static func write<T: Encodable>(
        _ value: T, filename: String
    ) throws {
        let documents = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(value)
        try data.write(
            to: documents.appendingPathComponent(filename),
            options: .atomic
        )
    }
}

/// Headless SwiftUI entry point for the one-pair signed phone runner.
@main
private struct A17ParityApp: App {
    /// Root scene kept foregrounded while GPU-permitted suffixes execute.
    var body: some Scene {
        WindowGroup {
            A17ParityView()
        }
    }
}
