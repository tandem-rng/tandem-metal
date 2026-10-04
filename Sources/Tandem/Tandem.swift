// Tandem8x32 on the CPU: the building blocks, keys and draws of https://github.com/tandem-rng/spec
// and the derived draws of its Appendix A. Copyright 2026 Jessica Cox. Apache License 2.0.

/// A Tandem8x32 generator: the transport form (key, bit position, chunk length K) plus a cache
/// of the eight chunk states of the current 1024-bit row. The cache is a pure function of the
/// transport form, so copies draw the same values and equality ignores it.
public struct Tandem: Sendable {
    public let key: SIMD4<UInt32>
    public let chunkLength: UInt32
    var pos: UInt64
    var lanes = Lanes()
    var row: UInt64 = 0
    var cached = false

    /// `chunkLength` is a power of two in [1, 65536]. `position` is a bit position below 2^63.
    public init(key: SIMD4<UInt32>, position: UInt64 = 0, chunkLength: UInt32 = 32) {
        precondition(chunkLength.nonzeroBitCount == 1 && chunkLength <= 65536, "K must be a power of two up to 65536")
        precondition(position < 1 << 63, "a start position must be below 2^63")
        self.key = key
        self.chunkLength = chunkLength
        pos = position
    }

    /// The key of a 128-bit seed, whitened as section 6 of the specification requires.
    public init(seed: UInt128, chunkLength: UInt32 = 32) {
        var o = SIMD4<UInt32>(0, 0, Domain.seed, 0)
        var h = SIMD4<UInt32>((0..<4).map { UInt32(truncatingIfNeeded: seed >> (32 * $0)) })
        Tandem.seedF(&o, &h)
        self.init(key: o, chunkLength: chunkLength)
    }

    /// The bit position of the next draw. Set it to move the generator, at the same cost at
    /// any distance.
    public var position: UInt64 {
        get { pos }
        set {
            precondition(newValue < 1 << 63, "a start position must be below 2^63")
            pos = newValue
        }
    }
}

extension Tandem: Hashable {
    public static func == (a: Tandem, b: Tandem) -> Bool {
        a.key == b.key && a.pos == b.pos && a.chunkLength == b.chunkLength
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(key)
        hasher.combine(pos)
        hasher.combine(chunkLength)
    }
}

enum Domain {
    static let stream: UInt32 = 0x9e37_79b9
    static let split: UInt32 = 0xbb67_ae85
    static let fork: UInt32 = 0xd251_1f53
    static let fold: UInt32 = 0xcd9e_8d57
    static let seed: UInt32 = 0xa54f_f53a
    static let auxStream: UInt32 = 0x94d0_49bb
}

let clockWeyl: UInt32 = 0x9e37_79b9
let roundConstants: [UInt32] = [
    0xd17c_c1b7, 0xa722_0a94, 0xfe13_abe8, 0xfa9a_6ee0, 0xedb1_4acc, 0x9e21_c820, 0xff28_b1d5, 0xef5d_e2b0,
]

@inline(__always) func rotl(_ x: UInt32, _ r: UInt32) -> UInt32 { (x &<< r) | (x &>> (32 &- r)) }

@inline(__always) func rotl(_ x: SIMD8<UInt32>, _ r: UInt32) -> SIMD8<UInt32> { (x &<< r) | (x &>> (32 &- r)) }

@inline(__always) func align(_ p: UInt64, _ w: UInt64) -> UInt64 { (p &+ w &- 1) & ~(w &- 1) }

// MARK: Building blocks

extension Tandem {
    /// The step T of section 3 on one state.
    public static func stepT(_ o: inout SIMD4<UInt32>, _ h: inout SIMD4<UInt32>) {
        let p0 = UInt64(o[0]) &* UInt64(h[0] | 1), p1 = UInt64(o[2]) &* UInt64(h[1] | 1)
        let lo0 = UInt32(truncatingIfNeeded: p0), hi0 = UInt32(truncatingIfNeeded: p0 &>> 32)
        let lo1 = UInt32(truncatingIfNeeded: p1), hi1 = UInt32(truncatingIfNeeded: p1 &>> 32)
        let n = SIMD4(o[1] ^ hi1 ^ lo1, rotl(lo1, 16) ^ h[2], o[3] ^ hi0 ^ lo0, rotl(lo0, 16) ^ h[3])
        let a = h[0] ^ rotl(h[1], 7), b = h[1] ^ rotl(h[2], 13), c = h[2] ^ rotl(h[3], 22)
        h = SIMD4((a &+ clockWeyl) ^ n[0], b, c, h[3] ^ rotl(a, 3))
        o = n
    }

    /// The seeding function F of section 3: eight rounds of T, a round constant, a swap.
    public static func seedF(_ o: inout SIMD4<UInt32>, _ h: inout SIMD4<UInt32>) {
        for rc in roundConstants {
            stepT(&o, &h)
            o[0] ^= rc
            swap(&o, &h)
        }
    }

    /// F(key, counter, domain, aux).
    public static func seedF(key: SIMD4<UInt32>, counter: UInt64, domain: UInt32, aux: UInt32)
        -> (o: SIMD4<UInt32>, h: SIMD4<UInt32>)
    {
        var o = SIMD4(UInt32(truncatingIfNeeded: counter), UInt32(truncatingIfNeeded: counter >> 32), domain, aux)
        var h = key
        seedF(&o, &h)
        return (o, h)
    }

    /// Block B(c, j): the exposed half of chunk c after j + 1 steps.
    public static func block(key: SIMD4<UInt32>, chunk: UInt64, step: UInt32) -> SIMD4<UInt32> {
        var (o, h) = seedF(key: key, counter: chunk, domain: Domain.stream, aux: Domain.auxStream)
        for _ in 0...step { stepT(&o, &h) }
        return o
    }
}

// MARK: Derived generators

extension Tandem {
    func child(_ counter: UInt64, _ domain: UInt32, _ aux: UInt32, hidden: Bool) -> Tandem {
        let (o, h) = Tandem.seedF(key: key, counter: counter, domain: domain, aux: aux)
        return Tandem(key: hidden ? h : o, chunkLength: chunkLength)
    }

    /// Child `index` from the key alone, at position 0.
    public func split(_ index: UInt64) -> Tandem { child(index >> 1, Domain.split, 0, hidden: index & 1 == 1) }

    /// The child for a purpose identifier, from the key alone, at position 0.
    public func purpose(_ id: UInt64) -> Tandem { child(id, Domain.fold, 0, hidden: false) }

    /// `n` children of the current block. The position moves past that block, also for n = 0.
    public mutating func fork(_ n: Int) -> [Tandem] {
        precondition(n >= 0 && n <= 1 << 33, "fork takes up to 2^33 children")
        let b = pos >> 7
        let children = (0..<UInt64(n)).map { i in
            child(b, Domain.fork, UInt32(truncatingIfNeeded: i >> 1), hidden: i & 1 == 1)
        }
        pos = (b + 1) << 7
        return children
    }
}

// MARK: Rows

/// The eight chunks of a group at one step, word-major: o0 holds word 0 of every lane. One T
/// over a row is then a few vector operations per word.
struct Lanes: Sendable {
    var o0 = SIMD8<UInt32>(), o1 = SIMD8<UInt32>(), o2 = SIMD8<UInt32>(), o3 = SIMD8<UInt32>()
    var h0 = SIMD8<UInt32>(), h1 = SIMD8<UInt32>(), h2 = SIMD8<UInt32>(), h3 = SIMD8<UInt32>()

    mutating func seed(_ key: SIMD4<UInt32>, group: UInt64) {
        for lane in 0..<8 {
            let (o, h) = Tandem.seedF(key: key, counter: 8 &* group &+ UInt64(lane), domain: Domain.stream,
                                      aux: Domain.auxStream)
            o0[lane] = o[0]; o1[lane] = o[1]; o2[lane] = o[2]; o3[lane] = o[3]
            h0[lane] = h[0]; h1[lane] = h[1]; h2[lane] = h[2]; h3[lane] = h[3]
        }
    }

    @inline(__always) mutating func step() {
        let p0 = SIMD8<UInt64>(truncatingIfNeeded: o0) &* SIMD8<UInt64>(truncatingIfNeeded: h0 | 1)
        let p1 = SIMD8<UInt64>(truncatingIfNeeded: o2) &* SIMD8<UInt64>(truncatingIfNeeded: h1 | 1)
        let lo0 = SIMD8<UInt32>(truncatingIfNeeded: p0), hi0 = SIMD8<UInt32>(truncatingIfNeeded: p0 &>> 32)
        let lo1 = SIMD8<UInt32>(truncatingIfNeeded: p1), hi1 = SIMD8<UInt32>(truncatingIfNeeded: p1 &>> 32)
        let n0 = o1 ^ hi1 ^ lo1, n1 = rotl(lo1, 16) ^ h2, n2 = o3 ^ hi0 ^ lo0, n3 = rotl(lo0, 16) ^ h3
        let a = h0 ^ rotl(h1, 7)
        h1 ^= rotl(h2, 13)
        h2 ^= rotl(h3, 22)
        h3 ^= rotl(a, 3)
        h0 = (a &+ clockWeyl) ^ n0
        o0 = n0; o1 = n1; o2 = n2; o3 = n3
    }

    @inline(__always) func word(_ w: UInt64, lane: Int) -> UInt32 {
        switch w {
        case 0: o0[lane]
        case 1: o1[lane]
        case 2: o2[lane]
        default: o3[lane]
        }
    }

    /// The 128 bytes of the row in stream order.
    @inline(__always) func store(_ out: UnsafeMutableRawPointer) {
        for lane in 0..<8 {
            let at = 16 * lane
            out.storeBytes(of: o0[lane], toByteOffset: at, as: UInt32.self)
            out.storeBytes(of: o1[lane], toByteOffset: at + 4, as: UInt32.self)
            out.storeBytes(of: o2[lane], toByteOffset: at + 8, as: UInt32.self)
            out.storeBytes(of: o3[lane], toByteOffset: at + 12, as: UInt32.self)
        }
    }
}

extension Tandem {
    /// Moves the cache to row r: forward steps inside the cached group, else a reseed.
    @inline(__always) mutating func load(row r: UInt64) {
        if cached && r == row { return }
        let shift = UInt64(chunkLength.trailingZeroBitCount)
        if cached && r > row && r >> shift == row >> shift {
            for _ in row..<r { lanes.step() }
        } else {
            lanes.seed(key, group: r >> shift)
            for _ in 0...(r & UInt64(chunkLength - 1)) { lanes.step() }
        }
        row = r
        cached = true
    }

    @inline(__always) mutating func word(at p: UInt64) -> UInt32 {
        load(row: p >> 10)
        return lanes.word((p >> 5) & 3, lane: Int((p >> 7) & 7))
    }

    /// The stream bytes [b0, b1) into out.
    mutating func fillBytes(_ out: UnsafeMutableRawPointer, from b0: UInt64, to b1: UInt64) {
        guard b0 < b1 else { return }
        withUnsafeTemporaryAllocation(byteCount: 128, alignment: 16) { tmp in
            var at = b0, dst = out
            while at < b1 {
                let start = at & ~127, lo = Int(at - start), hi = Int(min(b1 - start, 128))
                load(row: start >> 7)
                if lo == 0 && hi == 128 {
                    lanes.store(dst)
                } else {
                    lanes.store(tmp.baseAddress!)
                    dst.copyMemory(from: tmp.baseAddress! + lo, byteCount: hi - lo)
                }
                dst += hi - lo
                at = start + UInt64(hi)
            }
        }
    }

    /// The end of n aligned draws of w bits from the position, which must stay below 2^64.
    func fillEnd(count n: Int, width w: UInt64) -> (UInt64, UInt64) {
        let p0 = align(pos, w), bits = UInt64(n).multipliedReportingOverflow(by: w)
        let p1 = p0.addingReportingOverflow(bits.partialValue)
        precondition(pos <= p0 && !bits.overflow && !p1.overflow, "the fill ends past bit 2^64")
        return (p0, p1.partialValue)
    }

    /// n aligned draws of w bits into out, as the stream's bytes.
    mutating func fillRaw(_ out: UnsafeMutableRawPointer?, count n: Int, width w: UInt64) {
        let (p0, p1) = fillEnd(count: n, width: w)
        if n > 0 { fillBytes(out!, from: p0 / 8, to: p1 / 8) }
        pos = p1
    }
}
