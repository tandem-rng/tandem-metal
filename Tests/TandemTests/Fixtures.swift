import Foundation
import Metal
import Tandem

/// The fixtures of tandem-c and tandem-cuda, written by tools/gen_fixtures.sh. 64-bit values and
/// f64 bits are hex strings, f32 values are their bits.
struct Cross: Decodable {
    struct Row32: Decodable {
        let start: UInt64?
        let range: UInt32
        let end: UInt64?
        let values: [UInt32]
    }

    struct Row64: Decodable {
        let start: UInt64?
        let range: String
        let end: UInt64?
        let values: [String]
    }

    struct Pairs: Decodable {
        let end: UInt64
        let values: [Value]
    }

    struct Fill: Decodable {
        let start: UInt64
        let n: Int?
        let end: UInt64?
        let values: [Value]
    }

    /// A hex string of f64 bits or a number holding f32 bits.
    enum Value: Decodable {
        case hex(UInt64)
        case bits(UInt32)

        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let s = try? c.decode(String.self) { self = .hex(UInt64(s, radix: 16)!) } else { self = .bits(try c.decode(UInt32.self)) }
        }

        var f64: Double { if case let .hex(b) = self { Double(bitPattern: b) } else { .nan } }
        var f32: Float { if case let .bits(b) = self { Float(bitPattern: b) } else { .nan } }
    }

    let key: [UInt32]
    let belowU32: [Row32], belowU64: [Row64]
    let fillBelowU32: [Row32], fillBelowU64: [Row64]
    let cudaBelowU32: [Row32], cudaBelowU64: [Row64], cudaBelowU32At: [Row32], cudaBelowU64At: [Row64]
    let normalF64: Pairs, normalF32: Pairs
    let fillNormalF64: [Fill], fillNormalF32: [Fill]
    let exponentialF64: [Fill], exponentialF32: [Fill]
    let fillExponentialF64: [Fill], fillExponentialF32: [Fill]

    /// Every bounded-fill row of the two implementations, 32-bit and 64-bit.
    var fills32: [Row32] { fillBelowU32 + cudaBelowU32 + cudaBelowU32At }
    var fills64: [Row64] { fillBelowU64 + cudaBelowU64 + cudaBelowU64At }
}

func fixtureURL(_ name: String) -> URL {
    Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures")!
}

let cross: Cross = {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try! decoder.decode(Cross.self, from: Data(contentsOf: fixtureURL("cross.json")))
}()

let seed42Key = SIMD4<UInt32>(cross.key)
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
