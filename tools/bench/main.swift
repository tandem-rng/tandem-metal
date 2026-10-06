// GiB/s of the GPU fills into device memory, no readback, against MPSMatrixRandomPhilox, and of
// the CPU fills on one core against arc4random_buf and drand48 loops.
// Usage: swift run -c release tandem-bench [gpu|cpu]
import Darwin
import Metal
import MetalPerformanceShaders
import Tandem

let what = CommandLine.arguments.dropFirst().first ?? "all"
let gib = Double(1 << 30)

func show(_ name: String, _ log2n: Int, _ bytes: Int, _ seconds: Double, _ base: Double) {
    let rate = { (s: Double) in String(format: "%7.2f", Double(bytes) / s / gib) }
    print("\(name.padding(toLength: 28, withPad: " ", startingAt: 0)) 2^\(log2n)  \(rate(seconds)) GiB/s  baseline \(rate(base)) GiB/s")
}

/// MPS has Philox for uint32 words, float32 uniforms and float32 normals. A draw with no MPS
/// counterpart gets the uint32 words of the same byte count.
enum MPSDraw {
    case words, uniform, normal

    init(_ draw: GPUDraw) {
        switch draw {
        case .f32, .exponentialF32: self = .uniform
        case .normalF32: self = .normal
        default: self = .words
        }
    }

    func kernel(_ device: MTLDevice) -> MPSMatrixRandomPhilox {
        let (type, desc): (MPSDataType, MPSMatrixRandomDistributionDescriptor) = switch self {
        case .words: (.uInt32, .default())
        case .uniform: (.float32, .uniformDistributionDescriptor(withMinimum: 0, maximum: 1))
        case .normal: (.float32, .normalDistributionDescriptor(withMean: 0, standardDeviation: 1))
        }
        return MPSMatrixRandomPhilox(device: device, destinationDataType: type, seed: 42, distributionDescriptor: desc)
    }
}

if what != "cpu" {
    let k = try TandemKernels(device: MTLCreateSystemDefaultDevice()!)
    print(k.device.name)
    let draws: [(String, GPUDraw)] = [
        ("u32", .u32), ("u64", .u64), ("f32", .f32), ("u32 below 1000", .u32Below(range: 1000)),
        ("u32 below 1000 wide", .u32BelowWide(range: 1000)), ("u64 below 1000", .u64Below(range: 1000)),
        ("u32 below 2^31 + 1", .u32Below(range: 0x8000_0001)), ("normal f32", .normalF32), ("normal f32, odd start", .normalF32),
        ("exponential f32", .exponentialF32),
    ]
    // Many fills per command buffer, timed on the GPU, so the commit latency is spread.
    func best(_ encode: (MTLCommandBuffer) -> Void) -> Double {
        func run(_ reps: Int) -> Double {
            let cb = k.queue.makeCommandBuffer()!
            for _ in 0..<reps { encode(cb) }
            cb.commit()
            cb.waitUntilCompleted()
            return (cb.gpuEndTime - cb.gpuStartTime) / Double(reps)
        }
        _ = run(4)
        return (0..<7).map { _ in run(16) }.min()!
    }
    for log2n in [24, 26] {
        let n = 1 << log2n
        let buffer = k.device.makeBuffer(length: 8 * n, options: .storageModePrivate)!
        for (name, draw) in draws {
            // A start at an odd f32 draw runs the kernel that pairs draws across blocks.
            var r = Tandem(key: SIMD4(1, 2, 3, 4), position: name.hasSuffix("odd start") ? 32 : 0)
            let ours = best { r.fill(draw, count: n, buffer: buffer, kernels: k, commandBuffer: $0) }
            let mps = MPSDraw(draw), philox = mps.kernel(k.device)
            let words = n * draw.size / 4
            let vector = MPSVector(buffer: buffer, descriptor: MPSVectorDescriptor(length: words, dataType: mps == .words ? .uInt32 : .float32))
            let theirs = best { philox.encode(commandBuffer: $0, destinationVector: vector) }
            show(name, log2n, n * draw.size, ours, theirs)
        }
    }
}

if what != "gpu" {
    var load = [0.0, 0.0, 0.0]
    getloadavg(&load, 3)
    guard load[0] < 5 else {
        print("CPU bench skipped: 1-minute load \(load[0]) is 5 or more")
        exit(0)
    }
    let log2n = 22, n = 1 << log2n
    func best(_ body: () -> Void) -> Double {
        (0..<5).map { _ in
            let t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            body()
            return Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0) * 1e-9
        }.min()!
    }
    var r = Tandem(seed: 42)
    var u32 = [UInt32](repeating: 0, count: n), u64 = [UInt64](repeating: 0, count: n)
    var f32 = [Float](repeating: 0, count: n), f64 = [Double](repeating: 0, count: n)
    // The fastest generators Swift reaches: arc4random_buf for raw words, then libc's drand48,
    // which beats SystemRandomNumberGenerator and GameplayKit's sources by four times or more.
    func words<T>(_ a: inout [T]) {
        a.withUnsafeMutableBytes { arc4random_buf($0.baseAddress, $0.count) }
    }
    func loop<T>(_ a: inout [T], _ draw: (Double) -> T) {
        a.withUnsafeMutableBufferPointer { p in for i in p.indices { p[i] = draw(drand48()) } }
    }
    // Box-Muller pairs, as the Tandem normals.
    func pairs<T>(_ a: inout [T], _ f: (Double, Double) -> (T, T)) {
        a.withUnsafeMutableBufferPointer { p in
            for i in stride(from: 0, to: p.count, by: 2) { (p[i], p[i + 1]) = f(drand48(), drand48()) }
        }
    }
    show("cpu u32", log2n, 4 * n, best { r.fillU32(&u32) }, best { words(&u32) })
    show("cpu u64", log2n, 8 * n, best { r.fillU64(&u64) }, best { words(&u64) })
    show("cpu f32", log2n, 4 * n, best { r.fillF32(&f32) }, best { loop(&f32) { Float($0) } })
    show("cpu f64", log2n, 8 * n, best { r.fillF64(&f64) }, best { loop(&f64) { $0 } })
    show("cpu u32 below 1000", log2n, 4 * n, best { r.fillU32(&u32, below: 1000) }, best { loop(&u32) { UInt32($0 * 1000) } })
    show("cpu normal f64", log2n, 8 * n, best { r.fillNormalF64(&f64) }, best {
        pairs(&f64) { a, b in let s = (-2 * log(1 - a)).squareRoot(); return (s * cos(2 * .pi * b), s * sin(2 * .pi * b)) }
    })
    show("cpu normal f32", log2n, 4 * n, best { r.fillNormalF32(&f32) }, best {
        pairs(&f32) { a, b in
            let s = (-2 * logf(1 - Float(a))).squareRoot(), t = 2 * Float.pi * Float(b)
            return (s * cosf(t), s * sinf(t))
        }
    })
    show("cpu exponential f64", log2n, 8 * n, best { r.fillExponentialF64(&f64) }, best { loop(&f64) { -log(1 - $0) } })
    show("cpu exponential f32", log2n, 4 * n, best { r.fillExponentialF32(&f32) }, best { loop(&f32) { -logf(1 - Float($0)) } })
}
