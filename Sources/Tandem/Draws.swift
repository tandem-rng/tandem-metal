// Scalar draws and CPU fills: section 5 of the specification and the derived draws of its
// Appendix A. Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.

@inline(__always) func toF32(_ raw: UInt32) -> Float { Float(raw >> 8) * 0x1p-24 }
@inline(__always) func toF64(_ raw: UInt64) -> Double { Double(raw >> 11) * 0x1p-53 }

// MARK: Uniform draws

extension Tandem {
    /// Each draw aligns the position to its width, reads, and advances.
    public mutating func nextU32() -> UInt32 {
        let p = align(pos, 32)
        pos = p + 32
        return word(at: p)
    }

    public mutating func nextU64() -> UInt64 {
        let p = align(pos, 64)
        pos = p + 64
        return UInt64(word(at: p)) | UInt64(word(at: p + 32)) << 32
    }

    /// (raw >> 8) 2^-24 for a 32-bit draw.
    public mutating func nextF32() -> Float { toF32(nextU32()) }

    /// (raw >> 11) 2^-53 for a 64-bit draw.
    public mutating func nextF64() -> Double { toF64(nextU64()) }

    /// Fills: element i is draw i from the aligned position, the same values as scalar draws.
    /// An empty fill aligns the position.
    public mutating func fillU32(_ out: inout [UInt32]) {
        let n = out.count
        out.withUnsafeMutableBytes { fillRaw($0.baseAddress, count: n, width: 32) }
    }

    public mutating func fillU64(_ out: inout [UInt64]) {
        let n = out.count
        out.withUnsafeMutableBytes { fillRaw($0.baseAddress, count: n, width: 64) }
    }

    public mutating func fillF32(_ out: inout [Float]) {
        let n = out.count
        out.withUnsafeMutableBytes { fillRaw($0.baseAddress, count: n, width: 32) }
        for i in out.indices { out[i] = toF32(out[i].bitPattern) }
    }

    public mutating func fillF64(_ out: inout [Double]) {
        let n = out.count
        out.withUnsafeMutableBytes { fillRaw($0.baseAddress, count: n, width: 64) }
        for i in out.indices { out[i] = toF64(out[i].bitPattern) }
    }
}

// MARK: Bounded integers

/// Purposes reserved for the fallback generators of bounded fills, Appendix A.
let purposeBelow32: UInt64 = 0x42_4c57_3332
let purposeBelow64: UInt64 = 0x42_4c57_3634

@inline(__always) func threshold<T: FixedWidthInteger & UnsignedInteger>(_ range: T) -> T {
    range == 0 ? 0 : (0 &- range) % range
}

extension Tandem {
    /// Uniform on [0, range) by Lemire's multiply and reject, a rejected draw replaced by the
    /// next draw of the stream. A range of 0 gives 0 and consumes one draw.
    public mutating func nextU32(below range: UInt32) -> UInt32 {
        var m = UInt64(nextU32()) &* UInt64(range)
        if UInt32(truncatingIfNeeded: m) < range {
            let t = threshold(range)
            while UInt32(truncatingIfNeeded: m) < t { m = UInt64(nextU32()) &* UInt64(range) }
        }
        return UInt32(m >> 32)
    }

    public mutating func nextU64(below range: UInt64) -> UInt64 {
        var m = nextU64().multipliedFullWidth(by: range)
        if m.low < range {
            let t = threshold(range)
            while m.low < t { m = nextU64().multipliedFullWidth(by: range) }
        }
        return m.high
    }

    /// The fallback for a rejected draw of a bounded fill: split(g) of purpose(P_w) of the key at
    /// position 0, where g is the draw's index in the stream. A fill cut at any element then
    /// equals the whole fill.
    func fallback(_ purpose: UInt64, _ g: UInt64) -> Tandem {
        Tandem(key: key, chunkLength: chunkLength).purpose(purpose).split(g)
    }

    @inline(__always) func below32(_ x: UInt32, _ range: UInt32, _ t: UInt32, _ g: UInt64) -> UInt32 {
        let m = UInt64(x) &* UInt64(range)
        if UInt32(truncatingIfNeeded: m) >= t { return UInt32(m >> 32) }
        var r = fallback(purposeBelow32, g)
        while true {
            let m = UInt64(r.nextU32()) &* UInt64(range)
            if UInt32(truncatingIfNeeded: m) >= t { return UInt32(m >> 32) }
        }
    }

    @inline(__always) func below64(_ x: UInt64, _ range: UInt64, _ t: UInt64, _ g: UInt64) -> UInt64 {
        let m = x.multipliedFullWidth(by: range)
        if m.low >= t { return m.high }
        var r = fallback(purposeBelow64, g)
        while true {
            let m = r.nextU64().multipliedFullWidth(by: range)
            if m.low >= t { return m.high }
        }
    }

    /// low + a uniform value on [0, range), wrapping. Element i maps draw i of the u32 fill, so
    /// the fill consumes one draw per element, and a rejected draw retries on a fallback
    /// generator keyed by the draw's index in the stream. An empty fill leaves the position.
    public mutating func fillU32(_ out: inout [UInt32], below range: UInt32, low: UInt32 = 0) {
        guard !out.isEmpty else { return }
        let first = align(pos, 32) >> 5, t = threshold(range)
        fillU32(&out)
        for i in out.indices { out[i] = below32(out[i], range, t, first + UInt64(i)) &+ low }
    }

    /// The same 32-bit draws as `fillU32(_:below:low:)`, stored as 64-bit values.
    public mutating func fillU32(_ out: inout [UInt64], below range: UInt32, low: UInt64 = 0) {
        guard !out.isEmpty else { return }
        var draws = [UInt32](repeating: 0, count: out.count)
        let first = align(pos, 32) >> 5, t = threshold(range)
        fillU32(&draws)
        for i in out.indices { out[i] = UInt64(below32(draws[i], range, t, first + UInt64(i))) &+ low }
    }

    public mutating func fillU64(_ out: inout [UInt64], below range: UInt64, low: UInt64 = 0) {
        guard !out.isEmpty else { return }
        let first = align(pos, 64) >> 6, t = threshold(range)
        fillU64(&out)
        for i in out.indices { out[i] = below64(out[i], range, t, first + UInt64(i)) &+ low }
    }
}

// MARK: Normals and exponentials

extension Tandem {
    /// Box-Muller of two uniforms a and b: (r cos 2 pi b, r sin 2 pi b) with r = sqrt(-2 ln(1 - a)),
    /// the polynomial step of tandem-c, bit for bit.
    public mutating func nextNormalPairF64() -> (Double, Double) {
        let a = nextF64()
        return normalPair(a, nextF64())
    }

    public mutating func nextNormalPairF32() -> (Float, Float) {
        let a = nextF32()
        return normalPair(a, nextF32())
    }

    /// The cos half of a pair. It consumes two uniforms and equals element 0 of a fill.
    public mutating func nextNormalF64() -> Double { nextNormalPairF64().0 }

    public mutating func nextNormalF32() -> Float { nextNormalPairF32().0 }

    /// Pair j is elements 2j and 2j + 1 from uniforms 2j and 2j + 1. An odd count keeps the cos
    /// half of its last pair and still consumes both uniforms. An empty fill leaves the position.
    public mutating func fillNormalF64(_ out: inout [Double]) {
        let pairs = out.count / 2
        if pairs > 0 {
            out.withUnsafeMutableBytes { fillRaw($0.baseAddress, count: 2 * pairs, width: 64) }
            for j in 0..<pairs {
                let z = normalPair(toF64(out[2 * j].bitPattern), toF64(out[2 * j + 1].bitPattern))
                out[2 * j] = z.0
                out[2 * j + 1] = z.1
            }
        }
        if out.count % 2 == 1 { out[out.count - 1] = nextNormalF64() }
    }

    public mutating func fillNormalF32(_ out: inout [Float]) {
        let pairs = out.count / 2
        if pairs > 0 {
            out.withUnsafeMutableBytes { fillRaw($0.baseAddress, count: 2 * pairs, width: 32) }
            for j in 0..<pairs {
                let z = normalPair(toF32(out[2 * j].bitPattern), toF32(out[2 * j + 1].bitPattern))
                out[2 * j] = z.0
                out[2 * j + 1] = z.1
            }
        }
        if out.count % 2 == 1 { out[out.count - 1] = nextNormalF32() }
    }

    /// -ln(1 - u) of one uniform, with the polynomial logarithm of the normals.
    public mutating func nextExponentialF64() -> Double { 0.5 * neg2Log(1 - nextF64()) }

    public mutating func nextExponentialF32() -> Float { 0.5 * neg2Log(1 - nextF32()) }

    /// Element i from uniform i. An empty fill leaves the position.
    public mutating func fillExponentialF64(_ out: inout [Double]) {
        guard !out.isEmpty else { return }
        fillF64(&out)
        for i in out.indices { out[i] = 0.5 * neg2Log(1 - out[i]) }
    }

    public mutating func fillExponentialF32(_ out: inout [Float]) {
        guard !out.isEmpty else { return }
        fillF32(&out)
        for i in out.indices { out[i] = 0.5 * neg2Log(1 - out[i]) }
    }
}
