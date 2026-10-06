import Foundation
import Metal
import Tandem

/// A case of the spec's conformance files, copies of tandem-spec f420545 conformance/*.json.
/// Values, ranges, weights and tables are hex strings of their bits.
struct Case: Decodable {
    let id: String, kind: String
    let key: [String]
    let K: UInt32
    let start: UInt64
    let range: String?, weights: [String]?, capacity: String?, cut: [String]?, alias: [String]?
    let n: Int
    let values: [String]
    let end: UInt64?
    let rejected: Int?

    var rng: Tandem { Tandem(key: SIMD4(key.map { UInt32($0, radix: 16)! }), position: start, chunkLength: K) }
    var bits: [UInt64] { values.map(hex64) }
    var f64: [Double] { bits.map(Double.init(bitPattern:)) }
    var f32: [Float] { bits.map { Float(bitPattern: UInt32($0)) } }
    var table: ChoiceTable { try! ChoiceTable(weights: weights!.map { Double(bitPattern: hex64($0)) }) }
}

struct Hashes: Decodable {
    struct Stream: Decodable {
        let file: String, type: String
        let key: [String]
        let K: UInt32
        let n: Int
        let sha256: String
    }

    struct Dump: Decodable {
        struct Draw: Decodable { let kind: String, n: Int }
        let id: String
        let key: [String]
        let starts: [UInt64]
        let draws: [Draw]
        let sha256: String?, fnv1a: String
        let end: UInt64?
    }

    let streams: [Stream], dumps: [Dump]
}

func fixtureURL(_ name: String) -> URL {
    Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures")!
}

func conformance(_ name: String) -> [Case] {
    struct File: Decodable { let cases: [Case] }
    return try! JSONDecoder().decode(File.self, from: Data(contentsOf: fixtureURL("conformance/\(name).json"))).cases
}

let hashes = try! JSONDecoder().decode(Hashes.self, from: Data(contentsOf: fixtureURL("conformance/hashes.json")))

/// The case whose id ends with " name".
func named(_ file: String, _ name: String) -> Case { conformance(file).first { $0.id.hasSuffix(" " + name) }! }

let seed42Key = SIMD4<UInt32>(0x421d_21eb, 0x32d3_1777, 0x62e7_564b, 0xdf2b_df82)
let key1234 = SIMD4<UInt32>(1, 2, 3, 4)

func hex64(_ s: String) -> UInt64 { UInt64(s, radix: 16)! }

/// A stream dump of tandem-c's tests/data as elements of T.
func dump<T>(_ name: String, as _: T.Type) -> [T] {
    let data = try! Data(contentsOf: fixtureURL(name + ".bin"))
    return data.withUnsafeBytes { Array($0.bindMemory(to: T.self)) }
}

func bytes<T>(_ a: [T]) -> [UInt8] { a.withUnsafeBytes { Array($0) } }

/// FNV-1a over bytes, as in tandem-c's bit tests.
func fnv(_ h: UInt64, _ b: [UInt8]) -> UInt64 {
    b.reduce(h) { ($0 ^ UInt64($1)) &* 0x100_0000_01b3 }
}

// MARK: GPU

// No global MTLDevice: before the macOS 26 SDK it is not Sendable, which Swift 6 rejects.
let hasGPU = MTLCreateSystemDefaultDevice() != nil

/// The kernels of the default device. A compile failure fails the tests that use them.
let kernels: Result<TandemKernels, Error>? = MTLCreateSystemDefaultDevice().map { d in Result { try TandemKernels(device: d) } }

/// The CPU fill of the same kind and length as a GPU fill, as bytes.
func cpuFill(_ rng: inout Tandem, _ draw: GPUDraw, _ n: Int) -> [UInt8] {
    switch draw {
    case .u32: var a = [UInt32](repeating: 0, count: n); rng.fillU32(&a); return bytes(a)
    case .u64: var a = [UInt64](repeating: 0, count: n); rng.fillU64(&a); return bytes(a)
    case .f32: var a = [Float](repeating: 0, count: n); rng.fillF32(&a); return bytes(a)
    case let .u32Below(r, l): var a = [UInt32](repeating: 0, count: n); rng.fillU32(&a, below: r, low: l); return bytes(a)
    case let .u32BelowWide(r, l): var a = [UInt64](repeating: 0, count: n); rng.fillU32(&a, below: r, low: l); return bytes(a)
    case let .u64Below(r, l): var a = [UInt64](repeating: 0, count: n); rng.fillU64(&a, below: r, low: l); return bytes(a)
    case .normalF32: var a = [Float](repeating: 0, count: n); rng.fillNormalF32(&a); return bytes(a)
    case .exponentialF32: var a = [Float](repeating: 0, count: n); rng.fillExponentialF32(&a); return bytes(a)
    case let .choice(t): var a = [UInt32](repeating: 0, count: n); rng.fillChoice(&a, table: t); return bytes(a)
    }
}

/// A GPU fill of n elements at `offset` bytes into a buffer of sentinel bytes. Returns the
/// filled bytes, and whether the bytes around them kept the sentinel.
func gpuFill(_ k: TandemKernels, _ rng: inout Tandem, _ draw: GPUDraw, _ n: Int, offset: Int = 0)
    -> (bytes: [UInt8], untouched: Bool)
{
    let size = n * draw.size, length = offset + size + 64
    let buffer = k.device.makeBuffer(length: length, options: .storageModeShared)!
    memset(buffer.contents(), 0xa5, length)
    rng.fill(draw, count: n, buffer: buffer, offset: offset, kernels: k)
    let all = Array(UnsafeRawBufferPointer(start: buffer.contents(), count: length))
    let around = all[..<offset] + all[(offset + size)...]
    return (Array(all[offset..<offset + size]), around.allSatisfy { $0 == 0xa5 })
}

/// A Float32 normal or exponential fill on the GPU when there is one, else on the CPU.
func fill32(normal: Bool) throws -> (inout Tandem, Int) -> [UInt8] {
    let draw: GPUDraw = normal ? .normalF32 : .exponentialF32
    guard let k = try kernels?.get() else { return { cpuFill(&$0, draw, $1) } }
    return { gpuFill(k, &$0, draw, $1).bytes }
}
