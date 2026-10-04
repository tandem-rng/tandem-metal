import Metal
import Testing
import Tandem

/// tandem-c's tests/test_normal_bits.c and test_exponential_bits.c: FNV-1a over f64 and f32
/// fills from several positions of one generator must equal tandem-c's hashes. f64 runs on the
/// CPU. f32 runs on the GPU when there is one, else on the CPU.
func bitsHash(_ fill32: (inout Tandem, Int) -> [UInt8], normal: Bool) -> UInt64 {
    let n = normal ? 1_999_999 : 1_000_000
    var g = Tandem(seed: 2026 | 7 << 64), h: UInt64 = 0xcbf2_9ce4_8422_2325
    var f64 = [Double](repeating: 0, count: n)
    for start: UInt64 in [0, 1, 77, 12345, 1 << 30] {
        g.position = start
        if normal { g.fillNormalF64(&f64) } else { g.fillExponentialF64(&f64) }
        h = fnv(h, bytes(f64))
        h = fnv(h, fill32(&g, n))
    }
    return h
}

func fill32(normal: Bool) throws -> (inout Tandem, Int) -> [UInt8] {
    let draw: GPUDraw = normal ? .normalF32 : .exponentialF32
    guard let k = try kernels?.get() else { return { cpuFill(&$0, draw, $1) } }
    return { gpuFill(k, &$0, draw, $1).bytes }
}

@Test func normalBits() throws {
    #expect(bitsHash(try fill32(normal: true), normal: true) == 0x9414_e131_5e26_53be)
}

@Test func exponentialBits() throws {
    #expect(bitsHash(try fill32(normal: false), normal: false) == 0x47f8_f982_97d9_4ee2)
}
