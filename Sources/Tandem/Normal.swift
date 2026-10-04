// The polynomial logarithm and Box-Muller step of tandem-c, with the same operations and fused
// multiply-adds, so the values are bit for bit those of tandem-c and the Metal kernels.
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

/// The angle 2 pi b is cut at the nearest quarter turn q, which is exact, short series give cos
/// and sin on the rest, and q swaps them and sets their signs: odd q swaps, bit 1 of q negates
/// the sine and bit 1 of q + 1 the cosine.
@inline(__always) func normalPair(_ a: Double, _ b: Double) -> (Double, Double) {
    let r = neg2Log(1 - a).squareRoot()
    let q = Int64(b * 4.0 + 0.5)
    let f = fma(-Double(q), 0.25, b), th = f * 6.283185307179586, w = th * th
    let hs = fma(w, fma(w, fma(w, fma(w, fma(w, 1.5914650986900946e-10, -2.5051097984389413e-08),
             2.755731600073921e-06), -0.00019841269836630226), 0.008333333333330813), -0.16666666666666669)
    let hc = fma(w, fma(w, fma(w, fma(w, fma(w, 2.0665708703855164e-09, -2.7555858522576447e-07),
             2.480158263811954e-05), -0.0013888888882156126), 0.04166666666663108), -0.4999999999999997)
    let sn = th * fma(w, hs, 1.0), cs = fma(w, hc, 1.0)
    let qu = UInt64(q), sm = 0 &- (qu & 1)
    let sb = sn.bitPattern, cb = cs.bitPattern
    let xb = ((sb & sm) | (cb & ~sm)) ^ ((qu &+ 1) << 62 & 0x8000_0000_0000_0000)
    let yb = ((cb & sm) | (sb & ~sm)) ^ (qu << 62 & 0x8000_0000_0000_0000)
    return (r * Double(bitPattern: xb), r * Double(bitPattern: yb))
}

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
