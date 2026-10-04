import Metal
import Testing
import Tandem

/// tandem-c's tests/test_normal_bits.c and test_exponential_bits.c: FNV-1a over the fills of n
/// elements at several positions of one generator must equal tandem-c's hashes.
func bitsHash(_ n: Int, _ fill: (inout Tandem, Int) -> [UInt8]) -> UInt64 {
    var g = Tandem(seed: 2026 | 7 << 64), h: UInt64 = 0xcbf2_9ce4_8422_2325
    for start: UInt64 in [0, 1, 77, 12345, 1 << 30] {
        g.position = start
        h = fnv(h, fill(&g, n))
    }
    return h
}

/// f32 runs on the GPU when there is one, else on the CPU. f64 runs on the CPU.
func fill32(normal: Bool) throws -> (inout Tandem, Int) -> [UInt8] {
    let draw: GPUDraw = normal ? .normalF32 : .exponentialF32
    guard let k = try kernels?.get() else { return { cpuFill(&$0, draw, $1) } }
    return { gpuFill(k, &$0, draw, $1).bytes }
}

@Test func normalBitsF64() {
    let h = bitsHash(1_000_000) { g, n in
        var a = [Double](repeating: 0, count: n)
        g.fillNormalF64(&a)
        return bytes(a)
    }
    #expect(h == 0xa61c_fa84_4c85_f7c1)
}

@Test func normalBitsF32() throws {
    #expect(bitsHash(1_999_999, try fill32(normal: true)) == 0xaa1e_a656_ce73_a4fb)
}

@Test func exponentialBits() throws {
    let f32 = try fill32(normal: false)
    let h = bitsHash(1_000_000) { g, n in
        var a = [Double](repeating: 0, count: n)
        g.fillExponentialF64(&a)
        return bytes(a) + f32(&g, n)
    }
    #expect(h == 0x47f8_f982_97d9_4ee2)
}
