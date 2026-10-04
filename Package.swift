// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "tandem-metal",
    platforms: [.macOS(.v15)],
    products: [.library(name: "Tandem", targets: ["Tandem"])],
    targets: [
        .target(name: "Tandem"),
        .executableTarget(name: "tandem-bench", dependencies: ["Tandem"], path: "tools/bench"),
        .testTarget(name: "TandemTests", dependencies: ["Tandem"], resources: [.copy("Fixtures")]),
    ]
)
