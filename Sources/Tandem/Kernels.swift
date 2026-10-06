// Fills on a Metal device: the kernels of tandem.metal and the host side that places them in
// the stream. Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.

import Metal

/// The compiled kernels of tandem.metal for one device, and a queue for blocking fills.
public final class TandemKernels: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    let pipelines: [String: MTLComputePipelineState]

    static let threads = 256
    static let names = [
        "fill_u32", "fill_f32", "fill_exponential_f32", "fill_below32", "fill_below32_wide",
        "fill_below64", "fill_normal_f32", "fill_normal_f32_odd", "fill_choice",
    ]

    public enum Failure: Error {
        case noQueue
        case threadgroupTooSmall(String, Int)
    }

    /// Compiles the shader with fast math off, which the normals and exponentials need to
    /// match tandem-c bit for bit.
    public init(device: MTLDevice) throws {
        let options = MTLCompileOptions()
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        let library = try device.makeLibrary(source: shaderSource, options: options)
        var pipelines: [String: MTLComputePipelineState] = [:]
        for name in Self.names {
            let p = try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
            // The odd normal kernel shares words across a threadgroup of exactly this size.
            guard p.maxTotalThreadsPerThreadgroup >= Self.threads else {
                throw Failure.threadgroupTooSmall(name, p.maxTotalThreadsPerThreadgroup)
            }
            pipelines[name] = p
        }
        guard let queue = device.makeCommandQueue() else { throw Failure.noQueue }
        self.device = device
        self.queue = queue
        self.pipelines = pipelines
    }
}

/// What a GPU fill writes. Signed outputs are the same bits: pass a low bound as its bit pattern.
public enum GPUDraw: Sendable {
    /// Stream words, 4 bytes each.
    case u32
    /// 64-bit draws, 8 bytes each.
    case u64
    /// (raw >> 8) 2^-24 of 32-bit draws.
    case f32
    /// low + a value uniform on [0, range) from each 32-bit draw, Appendix A.
    case u32Below(range: UInt32, low: UInt32 = 0)
    /// The 32-bit bounded draws of `u32Below`, stored as 64-bit values.
    case u32BelowWide(range: UInt32, low: UInt64 = 0)
    /// low + a value uniform on [0, range) from each 64-bit draw.
    case u64Below(range: UInt64, low: UInt64 = 0)
    /// Box-Muller pairs of 32-bit uniforms in float, bit for bit those of tandem-c.
    case normalF32
    /// -ln(1 - u) of each 32-bit uniform, bit for bit that of tandem-c.
    case exponentialF32
    /// Indices of the alias table of Appendix C as UInt32, one 64-bit draw each, exact.
    case choice(ChoiceTable)

    /// Kernel, draw width, range, low bound, and whether an empty fill aligns the position.
    var layout: (String, UInt64, UInt64, UInt64, Bool) {
        switch self {
        case .u32: ("fill_u32", 32, 0, 0, true)
        case .u64: ("fill_u32", 64, 0, 0, true)
        case .f32: ("fill_f32", 32, 0, 0, true)
        case .exponentialF32: ("fill_exponential_f32", 32, 0, 0, false)
        case let .u32Below(r, l): ("fill_below32", 32, UInt64(r), UInt64(l), false)
        case let .u32BelowWide(r, l): ("fill_below32_wide", 32, UInt64(r), l, false)
        case let .u64Below(r, l): ("fill_below64", 64, r, l, false)
        case .normalF32: ("fill_normal_f32", 32, 0, 0, false)
        case let .choice(t): ("fill_choice", 64, UInt64(t.cut.count), 0, true)
        }
    }

    /// Bytes of one output element.
    public var size: Int {
        switch self {
        case .u32, .f32, .u32Below, .normalF32, .exponentialF32, .choice: 4
        case .u64, .u32BelowWide, .u64Below: 8
        }
    }
}

extension Tandem {
    /// Writes `count` elements at `offset` bytes into `buffer` and advances the position as the
    /// CPU fill of the same kind does. With a command buffer the fill is only encoded into it,
    /// else it runs on the kernels' queue and returns when done. `offset` is a multiple of the
    /// element size.
    public mutating func fill(_ draw: GPUDraw, count: Int, buffer: MTLBuffer, offset: Int = 0,
                              kernels: TandemKernels, commandBuffer: MTLCommandBuffer? = nil)
    {
        precondition(count >= 0 && offset >= 0 && offset % draw.size == 0, "offset must be a multiple of the element size")
        precondition(offset + count * draw.size <= buffer.length, "the fill does not fit the buffer")
        let K = UInt64(chunkLength)
        var words = [UInt32](repeating: 0, count: 24)
        func put(_ i: Int, _ v: UInt64) {
            words[i] = UInt32(truncatingIfNeeded: v)
            words[i + 1] = UInt32(truncatingIfNeeded: v >> 32)
        }
        for i in 0..<4 { words[i] = key[i] }
        put(20, UInt64(offset))
        words[22] = chunkLength

        let name: String, g0: UInt64, g1: UInt64
        if case .normalF32 = draw {
            guard count > 0 else { return }
            let (p0, p1) = fillEnd(count: count + count % 2, width: 32)
            let s0 = p0 >> 5, pairs = UInt64(count + 1) / 2
            let ba = s0 >> 2, bb = (s0 + 2 * pairs - 1) >> 2
            name = s0 & 1 == 0 ? "fill_normal_f32" : "fill_normal_f32_odd"
            g0 = (ba >> 3) / K
            g1 = (bb >> 3) / K
            put(16, UInt64(count))
            put(18, s0)
            pos = p1
        } else {
            let (kernel, w, range, low, plain) = draw.layout
            name = kernel
            // An empty fill leaves the position, except a uniform or choice fill, which aligns it.
            if count == 0 {
                if plain { pos = align(pos, w) }
                return
            }
            let (p0, p1) = fillEnd(count: count, width: w)
            let b0 = p0 / 8, b1 = p1 / 8
            g0 = (b0 >> 7) / K
            g1 = ((b1 - 1) >> 7) / K
            put(6, b0)
            put(8, b1)
            put(10, range)
            put(12, low)
            if case let .choice(t) = draw {
                put(14, t.capacity)
            } else {
                put(14, w == 32 ? UInt64(threshold(UInt32(range))) : threshold(range))
            }
            pos = p1
        }
        put(4, g0)

        let cb = commandBuffer ?? kernels.queue.makeCommandBuffer()!
        let encoder = cb.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(kernels.pipelines[name]!)
        encoder.setBytes(words, length: 4 * words.count, index: 0)
        encoder.setBuffer(buffer, offset: offset, index: 1)
        if case let .choice(t) = draw {
            let device = kernels.device
            encoder.setBuffer(device.makeBuffer(bytes: t.cut, length: 8 * t.cut.count)!, offset: 0, index: 2)
            encoder.setBuffer(device.makeBuffer(bytes: t.alias, length: 4 * t.alias.count)!, offset: 0, index: 3)
        }
        let threads = 8 * Int(g1 - g0 + 1), t = TandemKernels.threads
        encoder.dispatchThreadgroups(MTLSize(width: (threads + t - 1) / t, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: t, height: 1, depth: 1))
        encoder.endEncoding()
        if commandBuffer == nil {
            cb.commit()
            cb.waitUntilCompleted()
        }
    }
}
