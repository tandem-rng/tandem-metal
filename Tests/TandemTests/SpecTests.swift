import Foundation
import Testing
import Tandem

/// Section 8 of the specification, from the spec repository's vectors.json.
struct Vectors {
    let json = try! JSONSerialization.jsonObject(with: Data(contentsOf: fixtureURL("vectors.json"))) as! [String: Any]

    func words(_ a: Any?) -> SIMD4<UInt32> { SIMD4((a as! [String]).map { UInt32($0, radix: 16)! }) }
    subscript(_ k: String) -> [String: Any] { json[k] as! [String: Any] }
    func rows(_ k: String) -> [[String: Any]] { json[k] as! [[String: Any]] }
}

@Test func stepAndSeedFunction() {
    let v = Vectors()
    for t in v.rows("T") {
        var o = v.words(t["o"]), h = v.words(t["h"])
        Tandem.stepT(&o, &h)
        #expect(o == v.words(t["o_out"]) && h == v.words(t["h_out"]))
    }
    let key = v.words(v.json["key"])
    for f in v.rows("F") {
        let s = Tandem.seedF(key: key, counter: UInt64(f["counter"] as! Int), domain: 0x9e37_79b9, aux: 0x94d0_49bb)
        #expect(s.o == v.words(f["o"]) && s.h == v.words(f["h"]))
    }
}

@Test func streamWordsAndDraws() {
    let v = Vectors()
    let key = v.words(v.json["key"])
    for s in v.rows("stream_words") {
        var r = Tandem(key: key, position: 32 * UInt64(s["first_word"] as! Int))
        #expect(SIMD4((0..<4).map { _ in r.nextU32() }) == v.words(s["words"]))
    }
    let draws = v["draws_from_position_0"]
    for (i, x) in draws["Float64"] as! [String: Double] {
        var r = Tandem(key: key, position: 64 * UInt64(i)!)
        #expect(r.nextF64() == x)
    }
    for (i, x) in draws["Float32"] as! [String: Double] {
        var r = Tandem(key: key, position: 32 * UInt64(i)!)
        #expect(r.nextF32() == Float(x))
    }
}

@Test func derivedKeysAndSeed() {
    let v = Vectors()
    let parent = Tandem(key: v.words(v.json["key"]))
    let d = v["derived_keys"]
    #expect(parent.split(0).key == v.words(d["split_child_0"]))
    #expect(parent.split(1).key == v.words(d["split_child_1"]))
    #expect(parent.purpose(7).key == v.words(d["purpose_7"]))
    var p = parent
    #expect(p.fork(1)[0].key == v.words(d["fork_child_0_at_block_0"]))

    let s = v["seed_whitening"]
    var r = Tandem(seed: UInt128(s["seed"] as! Int))
    #expect(r.key == v.words(s["key"]))
    for (i, x) in s["Float64"] as! [String: Double] {
        r.position = 64 * UInt64(i)!
        #expect(r.nextF64() == x)
    }
    r.position = 0
    #expect(r.nextU32() == 0x05e8_0cec)
}

/// Children start at position 0 with the parent's K. Split and purpose keep the parent's
/// position, a fork moves it past the current block, also for no children.
@Test func childrenAndPositions() {
    var p = Tandem(key: key1234, position: 300, chunkLength: 8)
    #expect(p.split(5).position == 0 && p.split(5).chunkLength == 8 && p.position == 300)
    let kids = p.fork(3)
    #expect(kids.count == 3 && p.position == 384)
    #expect(Set(kids.map(\.key)).count == 3)
    _ = p.fork(0)
    #expect(p.position == 512)
    _ = p.fork(0)
    #expect(p.position == 640)
}

@Test func randomNumberGenerator() {
    var a = Tandem(seed: 9), b = a
    #expect(a.next() == b.nextU64() && a == b)
    let x = Int.random(in: 0..<10, using: &a)
    #expect((0..<10).contains(x) && a.position > b.position)
}

/// tandem-c's stream dumps, from position 0 and from unaligned positions that cut rows.
@Test(arguments: [("k1234_K32_u32", 32), ("k1234_K8_u32", 8)])
func dumpsU32(name: String, K: UInt32) {
    let want = dump(name, as: UInt32.self)
    for start in [0, 1, 37, 1000, want.count - 5] {
        var r = Tandem(key: key1234, position: 32 * UInt64(start), chunkLength: K)
        var got = [UInt32](repeating: 0, count: want.count - start)
        r.fillU32(&got)
        #expect(got[...] == want[start...] && r.position == 32 * UInt64(want.count))
    }
}

@Test func dumpsU64F32F64() {
    var r = Tandem(key: key1234, position: 64 * 3)
    let u64 = dump("k1234_K32_u64", as: UInt64.self)
    var got64 = [UInt64](repeating: 0, count: u64.count - 3)
    r.fillU64(&got64)
    #expect(got64[...] == u64[3...])

    let f32 = dump("seed42_K32_f32", as: Float.self)
    var s = Tandem(key: seed42Key)
    var gotF32 = [Float](repeating: 0, count: f32.count)
    s.fillF32(&gotF32)
    #expect(gotF32 == f32)
    s.position = 32 * 7
    #expect((0..<9).map { _ in s.nextF32() } == Array(f32[7..<16]))

    let f64 = dump("seed42_K32_f64", as: Double.self)
    s.position = 1
    var gotF64 = [Double](repeating: 0, count: f64.count - 1)
    s.fillF64(&gotF64)
    #expect(gotF64[...] == f64[1...])
}
