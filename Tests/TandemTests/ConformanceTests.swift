import CryptoKit
import Foundation
import Testing
@testable import Tandem

// The spec's conformance files and the items of its conformance/CHECKLIST.md. The port has no
// complex draws. A start position at or past 2^63 and a fill that reaches 2^64 stop the process
// by a precondition, which a test cannot catch, so those items are not run here.

let fillFiles = ["fill_below", "normal", "exponential", "choice"]

/// The end position of a case: its own, or the rule for the draws its fill consumes.
func end(_ c: Case) -> UInt64 {
    if let e = c.end { return e }
    let w: UInt64 = c.kind.hasSuffix("32") ? 32 : 64
    let draws = c.kind == "fill_normal_f32" ? 2 * ((c.n + 1) / 2) : c.n
    return align(c.start, w) + w * UInt64(draws)
}

/// The elements where a case is cut: pair boundaries only for Float32 normals.
func cuts(_ c: Case) -> [Int] {
    let at = c.kind == "fill_normal_f32" ? [2, 8, 20, (c.n - 1) & ~1] : [1, 7, 20, 21, c.n - 1]
    return Set(at.filter { $0 > 0 && $0 < c.n }).sorted()
}

/// The CPU fill of a case's kind, as bit patterns.
func cpuBits(_ c: Case, _ r: inout Tandem, _ n: Int) -> [UInt64] {
    switch c.kind {
    case "fill_below_u32":
        var a = [UInt32](repeating: 0, count: n)
        r.fillU32(&a, below: UInt32(hex64(c.range!)))
        return a.map(UInt64.init)
    case "fill_below_u64":
        var a = [UInt64](repeating: 0, count: n)
        r.fillU64(&a, below: hex64(c.range!))
        return a
    case "fill_normal_f64", "fill_exponential_f64":
        var a = [Double](repeating: 0, count: n)
        if c.kind == "fill_normal_f64" { r.fillNormalF64(&a) } else { r.fillExponentialF64(&a) }
        return a.map(\.bitPattern)
    case "fill_normal_f32", "fill_exponential_f32":
        var a = [Float](repeating: 0, count: n)
        if c.kind == "fill_normal_f32" { r.fillNormalF32(&a) } else { r.fillExponentialF32(&a) }
        return a.map { UInt64($0.bitPattern) }
    default:
        var a = [UInt32](repeating: 0, count: n)
        r.fillChoice(&a, table: c.table)
        return a.map(UInt64.init)
    }
}

/// The kinds whose scalar draw is element 0 of a fill. A scalar bounded draw retries on the next
/// draw, and a scalar Float32 normal consumes a pair.
let scalarKinds: Set = ["fill_normal_f64", "fill_exponential_f64", "fill_exponential_f32", "fill_choice"]

func scalarBits(_ c: Case, _ r: inout Tandem) -> UInt64 {
    switch c.kind {
    case "fill_normal_f64": r.nextNormalF64().bitPattern
    case "fill_exponential_f64": r.nextExponentialF64().bitPattern
    case "fill_exponential_f32": UInt64(r.nextExponentialF32().bitPattern)
    default: UInt64(r.nextChoice(c.table))
    }
}

/// The GPU draw of a case's kind. Metal has no Float64.
func gpuDraw(_ c: Case) -> GPUDraw? {
    switch c.kind {
    case "fill_below_u32": .u32Below(range: UInt32(hex64(c.range!)))
    case "fill_below_u64": .u64Below(range: hex64(c.range!))
    case "fill_normal_f32": .normalF32
    case "fill_exponential_f32": .exponentialF32
    case "fill_choice": .choice(c.table)
    default: nil
    }
}

func gpuBits(_ k: TandemKernels, _ d: GPUDraw, _ r: inout Tandem, _ n: Int) -> [UInt64] {
    let b = gpuFill(k, &r, d, n).bytes
    return b.withUnsafeBytes { d.size == 4 ? $0.bindMemory(to: UInt32.self).map(UInt64.init) : Array($0.bindMemory(to: UInt64.self)) }
}

/// Every case on the CPU, whole, cut and filled in pieces on one generator, and as scalar draws.
@Test(arguments: fillFiles) func cpuCases(file: String) {
    for c in conformance(file) {
        var r = c.rng
        #expect(cpuBits(c, &r, c.n) == c.bits && r.position == end(c), "\(c.id)")
        for cut in cuts(c) {
            var p = c.rng
            #expect(cpuBits(c, &p, cut) + cpuBits(c, &p, c.n - cut) == c.bits && p.position == end(c), "\(c.id) cut \(cut)")
        }
        if scalarKinds.contains(c.kind) {
            var s = c.rng
            #expect(c.bits.map { _ in scalarBits(c, &s) } == c.bits && s.position == (c.n > 0 ? end(c) : c.start), "\(c.id) scalar")
        }
    }
}

/// Every case of a kind Metal draws, on the GPU, whole and cut.
@Test(.enabled(if: hasGPU), arguments: fillFiles) func gpuCases(file: String) throws {
    let k = try gpu()
    for c in conformance(file) {
        guard let d = gpuDraw(c) else { continue }
        var r = c.rng
        #expect(gpuBits(k, d, &r, c.n) == c.bits && r.position == end(c), "\(c.id)")
        for cut in cuts(c) {
            var p = c.rng
            #expect(gpuBits(k, d, &p, cut) + gpuBits(k, d, &p, c.n - cut) == c.bits && p.position == end(c), "\(c.id) cut \(cut)")
        }
    }
}

/// Scalar bounded draws: a rejected draw retries on the next draw, which the end pins.
@Test func scalarBelow() {
    for c in conformance("below") {
        var r = c.rng
        let got = c.bits.map { _ in c.kind == "below_u32" ? UInt64(r.nextU32(below: UInt32(hex64(c.range!)))) : r.nextU64(below: hex64(c.range!)) }
        #expect(got == c.bits && r.position == c.end, "\(c.id)")
    }
}

@Test func fallbackByGlobalIndex() {
    for (file, a, b) in [("fill_below", "CROSS_BELOW32_AT[4]", "CROSS_BELOW32[4]"), ("fill_below", "CROSS_BELOW64_AT[6]", "CROSS_BELOW64[6]"),
                         ("normal", "CROSS_NORMAL[1]", "CROSS_NORMAL[0]"), ("choice", "CROSS_CHOICE[1]", "CROSS_CHOICE[0]")] {
        #expect(named(file, a).bits.prefix(63) == named(file, b).bits.dropFirst(), "\(a)")
    }
    #expect(named("fill_below", "CROSS_BELOW32_AT[4]").rejected! > 0)
}

/// The fills name their width: 32-bit draws into 64-bit outputs keep the 32-bit values.
@Test func widthFromRange() {
    let c32 = named("fill_below", "CROSS_BELOW32[3]"), c64 = named("fill_below", "CROSS_BELOW64[3]")
    var a = c32.rng, b = c64.rng
    var wide = [UInt64](repeating: 0, count: 64), u64 = wide
    a.fillU32(&wide, below: 1000)
    b.fillU64(&u64, below: 1000)
    #expect(wide == c32.bits && u64 == c64.bits && wide != u64)
    var z = Tandem(key: seed42Key)
    #expect(z.nextU32(below: 0) == 0 && z.position == 32)
}

@Test func float32Pairs() {
    let pairs = named("normal", "CROSS_NORMALF"), at32 = named("normal", "CROSS_NORMAL32[1]")
    #expect(pairs.bits.prefix(33) == at32.bits[...])
    #expect(named("normal", "CROSS_NORMAL32[2]").bits.prefix(31) == named("normal", "CROSS_NORMAL32[0]").bits[2..<33])
    var r = at32.rng
    #expect(UInt64(r.nextNormalF32().bitPattern) == at32.bits[0] && r.position == 32 + 64)
}

@Test func choiceTables() throws {
    for c in conformance("choice") {
        let t = c.table
        #expect(t.capacity == hex64(c.capacity!), "\(c.id)")
        if let cut = c.cut, let alias = c.alias {
            #expect(t.cut == cut.map(hex64) && t.alias == alias.map { UInt32($0, radix: 16)! }, "\(c.id)")
        }
    }
    let single = named("choice", "CROSS_CHOICE[3]")
    #expect(single.weights!.count == 1 && single.bits.allSatisfy { $0 == 0 })
    for w: [Double] in [[], [1, -1], [1, .nan], [1, .infinity], [0, -0.0]] {
        #expect(throws: ChoiceTable.Failure.self) { try ChoiceTable(weights: w) }
    }
}

func sha256(_ b: [UInt8]) -> String { SHA256.hash(data: b).map { String(format: "%02x", $0) }.joined() }

let streamTypes: [String: @Sendable (inout Tandem, Int) -> [UInt8]] = [
    "UInt32": { r, n in var a = [UInt32](repeating: 0, count: n); r.fillU32(&a); return bytes(a) },
    "UInt64": { r, n in var a = [UInt64](repeating: 0, count: n); r.fillU64(&a); return bytes(a) },
    "Float32": { r, n in var a = [Float](repeating: 0, count: n); r.fillF32(&a); return bytes(a) },
    "Float64": { r, n in var a = [Double](repeating: 0, count: n); r.fillF64(&a); return bytes(a) },
]

/// The stream dumps the port draws, and the copies of tandem-c's dumps in Fixtures.
@Test func streamHashes() throws {
    var drawn = 0
    for s in hashes.streams {
        let name = (s.file as NSString).lastPathComponent
        if let url = Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures") {
            #expect(sha256(Array(try Data(contentsOf: url))) == s.sha256, "\(name)")
        }
        guard let fill = streamTypes[s.type] else { continue }
        var r = Tandem(key: SIMD4(s.key.map { UInt32($0, radix: 16)! }), chunkLength: s.K)
        #expect(sha256(fill(&r, s.n)) == s.sha256, "\(name)")
        drawn += 1
    }
    #expect(drawn == 5)
}

/// tandem-c's long outputs: Float32 fills run on the GPU when there is one.
@Test func dumpHashes() throws {
    let normal32 = try fill32(normal: true), exponential32 = try fill32(normal: false)
    for d in hashes.dumps {
        var out: [UInt8] = []
        var r = Tandem(key: SIMD4(d.key.map { UInt32($0, radix: 16)! }))
        for start in d.starts {
            r.position = start
            for draw in d.draws {
                switch draw.kind {
                case "fill_normal_f64":
                    var a = [Double](repeating: 0, count: draw.n)
                    r.fillNormalF64(&a)
                    out += bytes(a)
                case "fill_exponential_f64":
                    var a = [Double](repeating: 0, count: draw.n)
                    r.fillExponentialF64(&a)
                    out += bytes(a)
                default:
                    out += (draw.kind == "fill_normal_f32" ? normal32 : exponential32)(&r, draw.n)
                }
            }
        }
        #expect(String(format: "%016llx", fnv(0xcbf2_9ce4_8422_2325, out)) == d.fnv1a, "\(d.id)")
        if let sha = d.sha256 { #expect(sha256(out) == sha, "\(d.id)") }
        if let e = d.end { #expect(r.position == e, "\(d.id)") }
    }
}

/// Reads at any position equal the sequential fill across blocks, rows and chunks of K = 8, and a
/// 64-bit draw at 2^63 - 1 aligns to 2^63.
@Test func blockAnd2to63Boundaries() {
    var w = Tandem(key: seed42Key, chunkLength: 8)
    var whole = [UInt32](repeating: 0, count: 3000)
    w.fillU32(&whole)
    for p in [3, 4, 31, 32, 255, 256, 2047, 2048, 2900] {
        var r = Tandem(key: seed42Key, position: 32 * UInt64(p), chunkLength: 8)
        var got = [UInt32](repeating: 0, count: 100)
        r.fillU32(&got)
        #expect(got[...] == whole[p..<p + 100], "\(p)")
    }
    var top = Tandem(key: seed42Key, position: 1 << 63 - 1)
    _ = top.nextU64()
    #expect(top.position == (1 as UInt64) << 63 + 64)
}
