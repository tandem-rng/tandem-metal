import Testing
@testable import Tandem

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

/// A Float64 normal fill cut at a missed element and just after it equals the whole fill and
/// the scalar draws, at two chunk lengths, since each miss keys its fallback by its global index.
@Test func normalF64Cuts() {
    let n = 3000
    for K: UInt32 in [32, 8] {
        let start = Tandem(key: key1234, position: 37, chunkLength: K)
        var raw = [UInt64](repeating: 0, count: n), whole = [Double](repeating: 0, count: n)
        var w = start, s = start
        w.fillU64(&raw)
        let misses = raw.indices.filter { raw[$0] >> 11 >= zigK[Int(raw[$0] & 1023)] }
        #expect(misses.count >= 3)
        w = start
        w.fillNormalF64(&whole)
        let cuts = [0, misses[0], misses[0] + 1, misses[2], n]
        var part = start
        for (a, b) in zip(cuts, cuts.dropFirst()) {
            var got = [Double](repeating: 0, count: b - a)
            part.fillNormalF64(&got)
            #expect(got == Array(whole[a..<b]), "K = \(K) [\(a), \(b))")
        }
        #expect(whole.map { _ in s.nextNormalF64() } == whole && s.position == w.position && part.position == w.position)
    }
}
