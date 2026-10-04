import Foundation
import Testing
import Tandem

/// Raw moments 1 to 4 of x within 5 standard errors, where m[k] = E[X^k] for k = 0 to 8 gives
/// Var(X^k) = m[2k] - m[k]^2.
func momentsAgree(_ x: [Double], _ m: [Double]) -> Bool {
    let n = Double(x.count)
    return (1...4).allSatisfy { k in
        let mean = x.reduce(0) { $0 + pow($1, Double(k)) } / n
        return abs(mean - m[k]) < 5 * ((m[2 * k] - m[k] * m[k]) / n).squareRoot()
    }
}

/// The Kolmogorov-Smirnov distance of x from the distribution `cdf`.
func ksDistance(_ x: [Double], _ cdf: (Double) -> Double) -> Double {
    let s = x.sorted(), n = Double(s.count)
    return s.indices.reduce(0) { d, i in
        let F = cdf(s[i])
        return max(d, F - Double(i) / n, Double(i + 1) / n - F)
    }
}

/// 10^7 draws of f64 on the CPU and f32 on the GPU, or the CPU without one. 1.95 / sqrt(n) is the
/// KS critical value at level 0.001.
func checkDistribution(normal: Bool) throws {
    let n = 10_000_000, critical = 1.95 / Double(n).squareRoot()
    var r = Tandem(seed: 5)
    var f64 = [Double](repeating: 0, count: n)
    if normal { r.fillNormalF64(&f64) } else { r.fillExponentialF64(&f64) }
    let f32 = try fill32(normal: normal)(&r, n).withUnsafeBytes { $0.bindMemory(to: Float.self).map(Double.init) }
    let m: [Double] = normal ? [1, 0, 1, 0, 3, 0, 15, 0, 105] : [1, 1, 2, 6, 24, 120, 720, 5040, 40320]
    let cdf: (Double) -> Double = normal ? { 0.5 * erfc(-$0 / 2.0.squareRoot()) } : { -expm1(-$0) }
    for x in [f64, f32] {
        #expect(momentsAgree(x, m))
        #expect(ksDistance(x, cdf) < critical)
    }
}

@Test func normalDistribution() throws { try checkDistribution(normal: true) }

@Test func exponentialDistribution() throws { try checkDistribution(normal: false) }
