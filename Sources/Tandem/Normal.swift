// The polynomial logarithm, the Float32 Box-Muller step and the Float64 ziggurat of tandem-c,
// with the same operations and fused multiply-adds, so the values are bit for bit those of
// tandem-c and the Metal kernels.
// Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
//
// Swift does not contract a * b + c, so every fused operation below is an explicit fma.

@inline(__always) func fma(_ a: Double, _ b: Double, _ c: Double) -> Double { c.addingProduct(a, b) }
@inline(__always) func fma(_ a: Float, _ b: Float, _ c: Float) -> Float { c.addingProduct(a, b) }

/// -2 ln x for x in (0, 1]. x = m 2^k with m in [sqrt(1/2), sqrt(2)) from the bits: adding the
/// bits of sqrt(1/2) to the exponent field makes the mantissa rollover pick k. Then
/// -2 ln x = 2 nk ln 2 - 4 s p with s = (m - 1) / (m + 1), p a series in s^2, and ln 2 split so
/// that nk ln2_hi is exact.
@inline(__always) func neg2Log(_ x: Double) -> Double {
    let ix = x.bitPattern &+ 0x0009_5f62_0000_0000
    let nk = Double(1023 - Int64(ix >> 52))
    let m = Double(bitPattern: (ix & 0x000f_ffff_ffff_ffff) &+ 0x3fe6_a09e_0000_0000)
    let s = (m - 1) / (m + 1), z = s * s
    let p = fma(z, fma(z, fma(z, fma(z, fma(z, fma(z, 0.08312363319426472, 0.09070001083303751),
            0.11111433317907482), 0.14285712049336274), 0.2000000000566491), 0.33333333333331017), 1.0)
    return fma(nk, 3.816429394731813e-10, fma(nk, 1.3862943607382476, (s * -4.0) * p))
}

@inline(__always) func neg2Log(_ x: Float) -> Float {
    let ix = x.bitPattern &+ 0x004a_fb0d
    let nk = Float(127 - Int32(ix >> 23))
    let m = Float(bitPattern: (ix & 0x007f_ffff) &+ 0x3f35_04f3)
    let s = (m - 1) / (m + 1), z = s * s
    let p = fma(z, fma(z, fma(z, 0.14275366, 0.20000061), 0.33333334), 1.0)
    return fma(nk, 2.857213530660374e-06, fma(nk, 1.38629150390625, (s * -4.0) * p))
}

/// -ln x for x in (0, 1], the Float32 exponential of tandem-c, within 0.58 ulp. The leading term
/// u = (2 - 2m) / (m + 1) is carried as uh + r / d, with m + 1 = d + dl exactly and r the
/// residual of uh, and nk ln2_hi + uh is split exactly by fast two-sum.
@inline(__always) func negLog(_ x: Float) -> Float {
    let ix = x.bitPattern &+ 0x004a_fb0d
    let nk = Float(127 - Int32(ix >> 23))
    let m = Float(bitPattern: (ix & 0x007f_ffff) &+ 0x3f35_04f3)
    let num = fma(m, -2.0, 2.0), d = m + 1, dl = m - (d - 1)
    let rcp = 1 / d, uh = fma(num, rcp, 0.0)
    let r = fma(-uh, dl, fma(-uh, d, num)), v = uh * uh
    let q = fma(v, fma(v, 0.0023109776, 0.012496489), 0.08333336)
    let a = nk * 0.693145751953125, hi = a + uh, e = uh - (hi - a)
    return hi + fma(uh * v, q, fma(r, rcp, fma(nk, 1.428606765330187e-06, e)))
}

/// The angle 2 pi b is cut at the nearest quarter turn q, which is exact, short series give cos
/// and sin on the rest, and q swaps them and sets their signs: odd q swaps, bit 1 of q negates
/// the sine and bit 1 of q + 1 the cosine.
@inline(__always) func normalPair(_ a: Float, _ b: Float) -> (Float, Float) {
    let r = neg2Log(1 - a).squareRoot()
    let q = Int32(b * 4.0 + 0.5)
    let f = fma(-Float(q), 0.25, b)
    // 2 pi as a float pair, so the angle is good to the last bit of the float.
    let th = fma(f, -1.7484555e-7, f * 6.2831855), w = th * th
    let hs = fma(w, fma(w, fma(w, 2.72499e-06, -0.00019840087), 0.008333332), -0.16666667)
    let hc = fma(w, fma(w, fma(w, 2.4463761e-05, -0.0013887589), 0.04166665), -0.5)
    let sn = th * fma(w, hs, 1.0), cs = fma(w, hc, 1.0)
    let qu = UInt32(q), sm = 0 &- (qu & 1)
    let sb = sn.bitPattern, cb = cs.bitPattern
    let xb = ((sb & sm) | (cb & ~sm)) ^ ((qu &+ 1) << 30 & 0x8000_0000)
    let yb = ((cb & sm) | (sb & ~sm)) ^ (qu << 30 & 0x8000_0000)
    return (r * Float(bitPattern: xb), r * Float(bitPattern: yb))
}

// MARK: Float64 ziggurat

/// The purpose reserved for the fallback generators of the Float64 normals, Appendix A.
let purposeNormal64: UInt64 = 0x4e_524d_3634

/// The 64-bit draws of a generator from position 0, one block at a time. A missed normal reads
/// a few draws, so it skips the eight-lane row cache of a `Tandem`.
struct BlockDraws {
    let key: SIMD4<UInt32>, chunkLength: UInt64
    var d: UInt64 = 0, block = SIMD4<UInt32>()

    /// Draw d is words 2 (d & 1) and 2 (d & 1) + 1 of block d >> 1, in row d >> 4.
    mutating func next() -> UInt64 {
        if d & 1 == 0 {
            let row = d >> 4
            block = Tandem.block(key: key, chunk: 8 * (row / chunkLength) + ((d >> 1) & 7),
                                 step: UInt32(row % chunkLength))
        }
        let w = 2 * Int(d & 1)
        d += 1
        return UInt64(block[w]) | UInt64(block[w + 1]) << 32
    }
}

/// The fast path for draw r: bits 0-9 pick the layer, bit 10 the sign and bits 11-63 the
/// magnitude. Returns nil when r misses the inner rectangle of its layer.
@inline(__always) func zigFast(_ r: UInt64) -> Double? {
    let ra = r >> 11
    return ra < zigK[Int(r & 1023)] ? Double(ra) * zigW[Int(r & 2047)] : nil
}

/// The slow path of Appendix A from a missed draw r, on the draws of its fallback generator. ln
/// is -0.5 neg2Log, which is exact given neg2Log, and every other operation rounds once.
@inline(never) func zigSlow(_ r0: UInt64, _ f: inout BlockDraws) -> Double {
    var r = r0
    while true {
        let i = Int(r & 1023), ra = r >> 11, x = Double(ra) * zigW[Int(r & 2047)]
        if ra < zigK[i] { return x }
        if i == 0 {
            // The tail beyond R, by Marsaglia's method.
            var a: Double, b: Double
            repeat {
                a = 0.5 * neg2Log(1 - toF64(f.next())) / zigR
                b = 0.5 * neg2Log(1 - toF64(f.next()))
            } while b + b < a * a
            return (r >> 10) & 1 == 1 ? -(zigR + a) : zigR + a
        }
        let y = zigY[i] + toF64(f.next()) * (zigY[i + 1] - zigY[i])
        if -0.5 * neg2Log(y) < -0.5 * (x * x) { return x }
        r = f.next()
    }
}
