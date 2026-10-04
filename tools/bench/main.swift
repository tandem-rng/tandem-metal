// GiB/s of the GPU fills into device memory, no readback, and of the CPU fills on one core.
// Usage: swift run -c release tandem-bench [gpu|cpu]
import Darwin
import Metal
import Tandem

let what = CommandLine.arguments.dropFirst().first ?? "all"
let gib = Double(1 << 30)

func show(_ name: String, _ log2n: Int, _ bytes: Int, _ seconds: Double) {
    print("\(name.padding(toLength: 28, withPad: " ", startingAt: 0)) 2^\(log2n)  \(String(format: "%7.1f", Double(bytes) / seconds / gib)) GiB/s")
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
    for log2n in [24, 26] {
        let n = 1 << log2n
        let buffer = k.device.makeBuffer(length: 8 * n, options: .storageModePrivate)!
        for (name, draw) in draws {
            // Many fills per command buffer, timed on the GPU, so the commit latency is spread.
            func run(_ reps: Int) -> Double {
                // A start at an odd f32 draw runs the kernel that pairs draws across blocks.
                var r = Tandem(key: SIMD4(1, 2, 3, 4), position: name.hasSuffix("odd start") ? 32 : 0)
                let cb = k.queue.makeCommandBuffer()!
                for _ in 0..<reps { r.fill(draw, count: n, buffer: buffer, kernels: k, commandBuffer: cb) }
                cb.commit()
                cb.waitUntilCompleted()
                return (cb.gpuEndTime - cb.gpuStartTime) / Double(reps)
            }
            _ = run(4)
            let best = (0..<7).map { _ in run(16) }.min()!
            show(name, log2n, n * draw.size, best)
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
    func best(_ bytes: Int, _ body: (inout Tandem) -> Void) -> Double {
        var r = Tandem(seed: 42)
        return (0..<5).map { _ in
            let t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            body(&r)
            return Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0) * 1e-9
        }.min()!
    }
    var u32 = [UInt32](repeating: 0, count: n), u64 = [UInt64](repeating: 0, count: n)
    var f32 = [Float](repeating: 0, count: n), f64 = [Double](repeating: 0, count: n)
    show("cpu u32", log2n, 4 * n, best(4 * n) { $0.fillU32(&u32) })
    show("cpu u64", log2n, 8 * n, best(8 * n) { $0.fillU64(&u64) })
    show("cpu f32", log2n, 4 * n, best(4 * n) { $0.fillF32(&f32) })
    show("cpu f64", log2n, 8 * n, best(8 * n) { $0.fillF64(&f64) })
    show("cpu u32 below 1000", log2n, 4 * n, best(4 * n) { $0.fillU32(&u32, below: 1000) })
    show("cpu normal f64", log2n, 8 * n, best(8 * n) { $0.fillNormalF64(&f64) })
    show("cpu normal f32", log2n, 4 * n, best(4 * n) { $0.fillNormalF32(&f32) })
    show("cpu exponential f64", log2n, 8 * n, best(8 * n) { $0.fillExponentialF64(&f64) })
    show("cpu exponential f32", log2n, 4 * n, best(4 * n) { $0.fillExponentialF32(&f32) })
}
