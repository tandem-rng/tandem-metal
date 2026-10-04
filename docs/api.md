# API

## Use

```swift
import Metal
import Tandem

let kernels = try TandemKernels(device: MTLCreateSystemDefaultDevice()!) // compiles the shader
var rng = Tandem(seed: 42)                     // 128-bit seed through the spec's whitening
let buffer = kernels.device.makeBuffer(length: 4 << 20, options: .storageModePrivate)!
rng.fill(.u32, count: 1 << 20, buffer: buffer, kernels: kernels)            // blocks until done
let cb = kernels.queue.makeCommandBuffer()!
rng.fill(.f32, count: 1 << 20, buffer: buffer, kernels: kernels, commandBuffer: cb) // only encodes
cb.commit()
rng.fill(.u32Below(range: 6), count: 1000, buffer: buffer, kernels: kernels)        // in [0, 6)
rng.fill(.u32BelowWide(range: 6, low: UInt64(bitPattern: -3)), count: 1000,        // Int64 in [-3, 3)
         buffer: buffer, kernels: kernels)
rng.fill(.normalF32, count: 1 << 20, buffer: buffer, kernels: kernels)
let worker = rng.split(7)                      // by index, from the key alone
let phase = rng.purpose(2)                     // by purpose identifier, from the key alone
let children = rng.fork(4)                     // from the current block; rng moves past it

var x = [Double](repeating: 0, count: 1000)    // the CPU path, f64 included
rng.fillNormalF64(&x)
let u = rng.nextF64(), k = rng.nextU64(below: 10), e = rng.nextExponentialF64()
```

## Reference

- `tandem.metal`: the building blocks `T`, `F`, `F_keyed`, `block`, `split_key`, `purpose_key`
  and the fills, as functions in namespace `tandem`, and the kernels `fill_u32`, `fill_f32`,
  `fill_below32`, `fill_below32_wide`, `fill_below64`, `fill_normal_f32`, `fill_normal_f32_odd`
  and `fill_exponential_f32`, which wrap them. One thread walks one chunk and stores each
  16-byte block of draws with one vector store. It is the one shader source of the Metal ports:
  [tandem-mlx](https://github.com/tandem-rng/tandem-mlx) vendors it unchanged and leaves the
  kernels out with `TANDEM_NO_KERNELS`.
- `Sources/Tandem`: the `Tandem` type, its CPU draws and fills, and `fill(_:count:buffer:...)`,
  which encodes a kernel. `Shader.swift` embeds `tandem.metal` (`tools/embed.sh`), so the
  package needs no resource bundle.

Metal has no double type. Float64 uniforms, normals and exponentials run on the CPU, and the
GPU fills `u64` words, which `(raw >> 11) 2^-53` maps to Float64 on the host.

A `Tandem` is the transport form of section 7, `key`, `position` and `chunkLength`, plus a cache
of the current row. Copies draw the same values, and equality compares the transport form. Set
`position` to move a generator, at the same cost at any distance.

`fill` takes a `GPUDraw`:

| case | element | values |
|---|---|---|
| `.u32`, `.u64` | 4, 8 bytes | stream words |
| `.f32` | 4 bytes | `(raw >> 8) 2^-24` |
| `.u32Below(range:low:)` | 4 bytes | `low` + Lemire on `[0, range)` from 32-bit draws |
| `.u32BelowWide(range:low:)` | 8 bytes | the same 32-bit draws, stored as 64-bit values |
| `.u64Below(range:low:)` | 8 bytes | `low` + Lemire on `[0, range)` from 64-bit draws |
| `.normalF32` | 4 bytes | Box-Muller pairs of f32 uniforms |
| `.exponentialF32` | 4 bytes | `-ln(1 - u)` of f32 uniforms |

`fill` writes `count` elements at `offset` bytes, a multiple of the element size, and moves the
position as the CPU fill of the same kind does. Without `commandBuffer` it runs on the kernels'
queue and returns when the fill is done. With one it only encodes, so fills can follow each
other in one command buffer, since each call returns with the new position. The low bound adds
in the output type with wrap-around, so a signed output takes the bit pattern of its bound.

The CPU draws are `nextU32`, `nextU64`, `nextF32`, `nextF64`, `nextU32(below:)`,
`nextU64(below:)`, `nextNormalF64`, `nextNormalPairF64`, `nextNormalF32`, `nextNormalPairF32`,
`nextExponentialF64` and `nextExponentialF32`. The CPU fills are `fillU32`, `fillU64`,
`fillF32`, `fillF64`, `fillU32(_:below:low:)` into 32-bit or 64-bit arrays,
`fillU64(_:below:low:)`, `fillNormalF64`, `fillNormalF32`, `fillExponentialF64` and
`fillExponentialF32`. `Tandem.stepT`, `Tandem.seedF` and `Tandem.block` expose the building
blocks for conformance tests. `Tandem` is a `RandomNumberGenerator` whose `next()` is
`nextU64()`, so `Int.random(in:using:)` and `shuffled(using:)` take it, with the standard
library's mappings rather than those of Appendix A.

## Parallel use

Element `i` of a fill is draw `i`, so threads, command buffers or devices that start at the
position of their first element, or draw from `split(task)`, reproduce a serial run for any
decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows. Start every range of a normal fill at an even element.
