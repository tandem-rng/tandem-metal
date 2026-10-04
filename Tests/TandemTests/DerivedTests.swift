import Testing
import Tandem

/// Scalar bounded draws of tandem-cuda's urand(range), from the seed-42 key at position 1. The
/// end position pins the number of rejected draws.
@Test func scalarBelow() {
    for row in cross.belowU32 {
        var r = Tandem(key: seed42Key, position: 1)
        #expect(row.values.map { _ in r.nextU32(below: row.range) } == row.values)
        #expect(r.position == row.end)
    }
    for row in cross.belowU64 {
        var r = Tandem(key: seed42Key, position: 1)
        #expect(row.values.map { _ in r.nextU64(below: hex64(row.range)) } == row.values.map(hex64))
        #expect(r.position == row.end)
    }
}

/// Bounded fills of tandem-c and tandem-cuda, rejected draws included, at starts where the
/// global draw index of element i differs from i.
@Test func fillBelow() {
    for row in cross.fills32 {
        var r = Tandem(key: seed42Key, position: row.start!)
        var got = [UInt32](repeating: 0, count: row.values.count)
        r.fillU32(&got, below: row.range)
        #expect(got == row.values, "start \(row.start!) range \(row.range)")
        #expect(r.position == row.end ?? align(row.start!, 32) + 32 * 64)
    }
    for row in cross.fills64 {
        var r = Tandem(key: seed42Key, position: row.start!)
        var got = [UInt64](repeating: 0, count: row.values.count)
        r.fillU64(&got, below: hex64(row.range))
        #expect(got == row.values.map(hex64), "start \(row.start!) range \(row.range)")
        #expect(r.position == row.end ?? align(row.start!, 64) + 64 * 64)
    }
}

func align(_ p: UInt64, _ w: UInt64) -> UInt64 { (p + w - 1) / w * w }

/// A low bound wraps in the output type, and a 32-bit draw widened to 64 bits keeps its value.
@Test func belowLowAndWide() {
    let start = Tandem(key: seed42Key, position: 12345)
    var a = start, b = start, c = start
    var plain = [UInt32](repeating: 0, count: 300), low = plain, wide = [UInt64](repeating: 0, count: 300)
    a.fillU32(&plain, below: 0xc000_0001)
    b.fillU32(&low, below: 0xc000_0001, low: 0x8000_0000)
    c.fillU32(&wide, below: 0xc000_0001, low: 1 << 40)
    #expect(low == plain.map { $0 &+ 0x8000_0000 })
    #expect(wide == plain.map { UInt64($0) + 1 << 40 })
    #expect(a == b && a == c)
}

/// A bounded fill cut at any element boundary equals the whole fill, because the fallback of a
/// rejected draw is keyed by its index in the stream. Half of these draws reject.
@Test(arguments: [1, 12345, 100_000] as [UInt64])
func fillBelowCut(start: UInt64) {
    let n = 300
    for cut in [1, 7, 100, 299] {
        let whole0 = Tandem(key: key1234, position: start)
        var whole = whole0, part = whole0
        var w = [UInt32](repeating: 0, count: n), p1 = [UInt32](repeating: 0, count: cut)
        var p2 = [UInt32](repeating: 0, count: n - cut)
        whole.fillU32(&w, below: 0xc000_0001)
        part.fillU32(&p1, below: 0xc000_0001)
        part.fillU32(&p2, below: 0xc000_0001)
        #expect(w == p1 + p2 && whole == part)

        var w64 = [UInt64](repeating: 0, count: n), q1 = [UInt64](repeating: 0, count: cut)
        var q2 = [UInt64](repeating: 0, count: n - cut)
        whole = whole0
        part = whole0
        whole.fillU64(&w64, below: 0xc000_0000_0000_0001)
        part.fillU64(&q1, below: 0xc000_0000_0000_0001)
        part.fillU64(&q2, below: 0xc000_0000_0000_0001)
        #expect(w64 == q1 + q2 && whole == part)
    }
}

/// Box-Muller pairs of tandem-c from position 1, bit for bit, and the scalar normal is the cos half.
@Test func normalPairs() {
    var r = Tandem(key: seed42Key, position: 1)
    let f64 = (0..<64).flatMap { _ in let z = r.nextNormalPairF64(); return [z.0, z.1] }
    #expect(f64.map(\.bitPattern) == cross.normalF64.values.map(\.f64.bitPattern) && r.position == cross.normalF64.end)
    r.position = 1
    let f32 = (0..<64).flatMap { _ in let z = r.nextNormalPairF32(); return [z.0, z.1] }
    #expect(f32.map(\.bitPattern) == cross.normalF32.values.map(\.f32.bitPattern) && r.position == cross.normalF32.end)
    r.position = 1
    #expect(r.nextNormalF64() == f64[0] && r.position == 192)
    r.position = 1
    #expect(r.nextNormalF32() == f32[0] && r.position == 96)
}

/// tandem-cuda's normal and exponential fills at start positions, bit for bit, with odd lengths.
@Test func fillNormalAndExponential() {
    for row in cross.fillNormalF64 {
        var r = Tandem(key: seed42Key, position: row.start)
        var got = [Double](repeating: 0, count: row.n!)
        r.fillNormalF64(&got)
        #expect(got.map(\.bitPattern) == row.values.map(\.f64.bitPattern), "start \(row.start)")
        #expect(r.position == align(row.start, 64) + 64 * UInt64(row.n! + 1))
    }
    for row in cross.fillNormalF32 {
        var r = Tandem(key: seed42Key, position: row.start)
        var got = [Float](repeating: 0, count: row.n!)
        r.fillNormalF32(&got)
        #expect(got.map(\.bitPattern) == row.values.map(\.f32.bitPattern), "start \(row.start)")
    }
    for row in cross.fillExponentialF64 {
        var r = Tandem(key: seed42Key, position: row.start)
        var got = [Double](repeating: 0, count: row.n!)
        r.fillExponentialF64(&got)
        #expect(got.map(\.bitPattern) == row.values.map(\.f64.bitPattern), "start \(row.start)")
    }
    for row in cross.fillExponentialF32 {
        var r = Tandem(key: seed42Key, position: row.start)
        var got = [Float](repeating: 0, count: row.n!)
        r.fillExponentialF32(&got)
        #expect(got.map(\.bitPattern) == row.values.map(\.f32.bitPattern), "start \(row.start)")
    }
}

/// tandem-c's exponential fills of 64, which equal 64 scalar draws, with the end position.
@Test func exponentials() {
    for row in cross.exponentialF64 {
        var a = Tandem(key: seed42Key, position: row.start), b = a
        var got = [Double](repeating: 0, count: 64)
        a.fillExponentialF64(&got)
        #expect(got.map(\.bitPattern) == row.values.map(\.f64.bitPattern) && a.position == row.end)
        #expect(got.map { _ in b.nextExponentialF64() } == got && b.position == row.end)
    }
    for row in cross.exponentialF32 {
        var a = Tandem(key: seed42Key, position: row.start), b = a
        var got = [Float](repeating: 0, count: 64)
        a.fillExponentialF32(&got)
        #expect(got.map(\.bitPattern) == row.values.map(\.f32.bitPattern) && a.position == row.end)
        #expect(got.map { _ in b.nextExponentialF32() } == got && b.position == row.end)
    }
}

/// An empty derived fill leaves the position, an empty uniform fill aligns it.
@Test func emptyFills() {
    for p: UInt64 in [1, 5, 33, 65, 1001] {
        var r = Tandem(key: key1234, position: p)
        var u32: [UInt32] = [], u64: [UInt64] = [], f32: [Float] = [], f64: [Double] = []
        r.fillU32(&u32, below: 10)
        r.fillU32(&u64, below: 10)
        r.fillU64(&u64, below: 10)
        r.fillNormalF64(&f64)
        r.fillNormalF32(&f32)
        r.fillExponentialF64(&f64)
        r.fillExponentialF32(&f32)
        #expect(r.position == p)
        r.fillU32(&u32)
        #expect(r.position == align(p, 32))
        r.fillF64(&f64)
        #expect(r.position == align(p, 64))
    }
}
