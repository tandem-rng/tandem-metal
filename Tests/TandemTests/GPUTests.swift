import Testing
import Tandem

/// The kernels compile, or the test fails. Without a Metal device the GPU tests are skipped.
func gpu() throws -> TandemKernels { try kernels!.get() }

/// tandem-c's dumps, from starts that cut rows and groups, into outputs at offsets that misalign
/// the 16-byte stores. Bytes outside the fill keep their sentinel.
@Test(.enabled(if: hasGPU), arguments: [("k1234_K32_u32", 32), ("k1234_K8_u32", 8)])
func gpuDumpsU32(name: String, K: UInt32) throws {
    let k = try gpu()
    let want = bytes(dump(name, as: UInt32.self))
    for (start, offset) in [(0, 0), (1, 0), (37, 4), (1000, 8), (16380, 12)] {
        var r = Tandem(key: key1234, position: 32 * UInt64(start), chunkLength: K)
        let n = want.count / 4 - start
        let got = gpuFill(k, &r, .u32, n, offset: offset)
        #expect(got.bytes[...] == want[(4 * start)...] && got.untouched, "start \(start)")
        #expect(r.position == 32 * UInt64(want.count / 4))
    }
}

@Test(.enabled(if: hasGPU)) func gpuDumpsU64F32() throws {
    let k = try gpu()
    let u64 = bytes(dump("k1234_K32_u64", as: UInt64.self))
    var r = Tandem(key: key1234, position: 65)
    let got = gpuFill(k, &r, .u64, u64.count / 8 - 2, offset: 8)
    #expect(got.bytes[...] == u64[16...] && got.untouched)

    let f32 = bytes(dump("seed42_K32_f32", as: Float.self))
    var s = Tandem(key: seed42Key, position: 32 * 3)
    let gotF32 = gpuFill(k, &s, .f32, f32.count / 4 - 3, offset: 4)
    #expect(gotF32.bytes[...] == f32[12...] && gotF32.untouched)
}

let gpuDraws: [GPUDraw] = [
    .u32, .u64, .f32, .u32Below(range: 1000), .u32Below(range: 0xc000_0001, low: 0x8000_0000),
    .u32BelowWide(range: 0xc000_0001, low: 1 << 40), .u64Below(range: 1_000_000),
    .u64Below(range: 0xc000_0000_0000_0001, low: 5), .u32Below(range: 0), .normalF32, .exponentialF32,
    .choice(try! ChoiceTable(weights: [3, 0, 1, 7.5, 0.125])), .choice(try! ChoiceTable(weights: [2])),
]

/// Every GPU fill equals the CPU fill of the same kind: bytes, end position, and no store
/// outside the fill. The starts give even and odd first draws, the lengths odd counts and fills
/// that span threadgroups, the offsets unaligned vector stores.
@Test(.enabled(if: hasGPU), arguments: 0..<gpuDraws.count)
func gpuEqualsCPU(draw index: Int) throws {
    let k = try gpu(), draw = gpuDraws[index]
    for K: UInt32 in [32, 8, 1] {
        for start: UInt64 in [0, 1, 32, 96, 160, 12345, 1 << 40 + 7] {
            for n in [1, 2, 3, 6, 33, 1001, 70001] {
                for offset in [0, draw.size] {
                    let r0 = Tandem(key: key1234, position: start, chunkLength: K)
                    var a = r0, b = r0
                    let want = cpuFill(&a, draw, n)
                    let got = gpuFill(k, &b, draw, n, offset: offset)
                    #expect(got.bytes == want && got.untouched && a == b, "\(draw) K \(K) start \(start) n \(n) offset \(offset)")
                }
            }
        }
    }
}

/// A GPU bounded fill cut at element boundaries equals the whole fill, rejections included.
@Test(.enabled(if: hasGPU)) func gpuFillBelowCut() throws {
    let k = try gpu()
    for draw: GPUDraw in [.u32Below(range: 0xc000_0001), .u64Below(range: 0xc000_0000_0000_0001)] {
        for start: UInt64 in [1, 12345] {
            for cut in [1, 7, 100, 299] {
                var whole = Tandem(key: key1234, position: start), part = whole
                let w = gpuFill(k, &whole, draw, 300).bytes
                let p = gpuFill(k, &part, draw, cut).bytes + gpuFill(k, &part, draw, 300 - cut).bytes
                #expect(w == p && whole == part, "\(draw) start \(start) cut \(cut)")
            }
        }
    }
}

/// An empty GPU fill writes nothing. It leaves the position, except a uniform or choice fill
/// aligns it.
@Test(.enabled(if: hasGPU)) func gpuEmptyFills() throws {
    let k = try gpu()
    for draw in gpuDraws {
        var a = Tandem(key: key1234, position: 33), b = a
        let got = gpuFill(k, &a, draw, 0)
        _ = cpuFill(&b, draw, 0)
        #expect(got.untouched && a == b, "\(draw)")
    }
}
