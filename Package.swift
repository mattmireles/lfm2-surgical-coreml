// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "LFM2SurgicalCoreML",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .library(
            name: "LFM2SurgicalRuntime",
            targets: ["LFM2SurgicalRuntime"]
        ),
        .executable(
            name: "lfm2-surgical-benchmark",
            targets: ["LFM2SurgicalBenchmark"]
        ),
        .executable(
            name: "lfm2-selective-benchmark",
            targets: ["LFM2SelectiveBenchmark"]
        ),
    ],
    targets: [
        .target(name: "LFM2SurgicalRuntime"),
        .executableTarget(
            name: "LFM2SurgicalBenchmark",
            dependencies: ["LFM2SurgicalRuntime"]
        ),
        .executableTarget(
            name: "LFM2SelectiveBenchmark",
            dependencies: ["LFM2SurgicalRuntime"]
        ),
    ]
)
