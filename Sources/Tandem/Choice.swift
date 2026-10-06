// Weighted choice of Appendix C of https://github.com/tandem-rng/spec: an alias table built from
// the weights in exact integer arithmetic. Copyright 2026 Jessica Cox. Apache License 2.0.

/// The integer alias table of Appendix C: column capacity `S`, and `cut` and `alias` per index.
/// Index i has probability proportional to its weight, and the table is the same in every port.
public struct ChoiceTable: Sendable, Equatable {
    public let capacity: UInt64
    public let cut: [UInt64]
    public let alias: [UInt32]

    public enum Failure: Error {
        /// No weights, 2^32 or more, a negative, infinite or NaN weight, or all weights zero.
        case invalidWeights
    }

    /// The table of `weights`, finite and nonnegative, not all zero, at most 2^32 - 1 of them.
    public init(weights: [Double]) throws {
        let m = weights.count
        guard m >= 1, m < 1 << 32, weights.allSatisfy({ $0.isFinite && $0 >= 0 }),
              let top = weights.max(), top > 0
        else { throw Failure.invalidWeights }
        let nbits = { (x: UInt64) in 64 - x.leadingZeroBitCount }
        let (topMant, topExp) = Self.parts(top)
        let e = topExp + nbits(topMant) - 1
        let t0 = 63 - nbits(UInt64(m)) - e
        let t = t0 + 63 - nbits(weights.reduce(0) { $0 + Self.ceil2($1, t0) })
        var cut = weights.map { Self.ceil2($0, t) }
        let total = cut.reduce(0, +), mm = UInt64(m)
        let pad = (mm - total % mm) % mm
        var b = 0
        for i in cut.indices where cut[i] > cut[b] { b = i }
        cut[b] += pad
        let s = (total + pad) / mm

        var alias = (0..<m).map { UInt32($0) }
        func full(_ from: Int) -> Int {
            var i = from
            while i < m && cut[i] < s { i += 1 }
            return i
        }
        var big = full(0)
        for i in 0..<m {
            var j = i
            while j <= i && cut[j] < s {
                alias[j] = UInt32(big)
                cut[big] -= s - cut[j]
                j = big
                if cut[big] < s { big = full(big + 1) }
            }
        }
        capacity = s
        self.cut = cut
        self.alias = alias
    }

    /// x = mant 2^exp with mant an integer below 2^53.
    static func parts(_ x: Double) -> (UInt64, Int) {
        let e = Int(x.exponentBitPattern)
        return e == 0 ? (x.significandBitPattern, -1074) : (x.significandBitPattern | 1 << 52, e - 1075)
    }

    /// The smallest integer not below x 2^t, exact. The callers keep it below 2^64.
    static func ceil2(_ x: Double, _ t: Int) -> UInt64 {
        let (mant, exp) = parts(x)
        let s = exp + t
        if mant == 0 { return 0 }
        if s >= 0 { return mant << UInt64(s) }
        if s <= -64 { return 1 }
        let sh = UInt64(-s)
        return (mant >> sh) + (mant & ((1 << sh) - 1) == 0 ? 0 : 1)
    }

    /// The index of a 64-bit draw.
    @inline(__always) func index(_ r: UInt64) -> UInt32 {
        let x = r.multipliedFullWidth(by: UInt64(cut.count))
        let j = Int(x.high)
        return x.low.multipliedFullWidth(by: capacity).high < cut[j] ? UInt32(j) : alias[j]
    }
}

extension Tandem {
    /// An index drawn by `table` from one 64-bit draw. It equals element 0 of a fill.
    public mutating func nextChoice(_ table: ChoiceTable) -> UInt32 { table.index(nextU64()) }

    /// Element i maps 64-bit draw i and never retries, so a fill cut at any element equals the
    /// whole fill. An empty fill aligns the position to 64 bits.
    public mutating func fillChoice(_ out: inout [UInt32], table: ChoiceTable) {
        var raw = [UInt64](repeating: 0, count: out.count)
        fillU64(&raw)
        for i in out.indices { out[i] = table.index(raw[i]) }
    }
}
