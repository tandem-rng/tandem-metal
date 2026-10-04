# tandem-metal

[Tandem8x32](https://github.com/tandem-rng/spec) for Metal: a noncryptographic pseudorandom
number generator built to be fast on CPUs and GPUs alike. Metal kernels fill GPU buffers, and a
Swift package places the fills in the stream, derives keys, and draws on the CPU. It produces
the stream the specification defines, bit for bit, and the bounded integers, normals and
exponentials of its Appendix A bit for bit as tandem-c does.

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

## Use

```swift
// Package.swift
.package(url: "https://github.com/tandem-rng/tandem-metal", branch: "main")
```

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
blocks for conformance tests.

Bounded integers, normals and exponentials follow Appendix A of the specification. A bounded
fill maps element `i` to draw `i`. A draw that Lemire rejects retries on `split(g)` of
`purpose(P_w)` of the key, where `g` is the draw's index in the stream, so a fill cut at any
element boundary equals the whole fill. A scalar bounded draw rejects onto the next draws of the
stream instead. Normal pair `j` is elements `2j` and `2j + 1` from uniforms `2j` and `2j + 1`,
so an odd count consumes one uniform more than it writes. An empty bounded, normal or
exponential fill leaves the position, and an empty uniform fill aligns it.

The normals and exponentials copy the arithmetic of tandem-c: a short series for `log` on the
exponent-split argument, and series for `cos` and `sin` on an angle cut at the nearest quarter
turn, with every multiply-add an explicit `fma`. The shader is compiled with fast math off and
takes division and square root from `precise::`, which round correctly. The GPU f32 normals and
exponentials and the CPU f32 and f64 ones are then bit for bit those of tandem-c.

Parallel use: element `i` of a fill is draw `i`, so threads, command buffers or devices that
start at the position of their first element, or draw from `split(task)`, reproduce a serial
run for any decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows. Start every range of a normal fill at an even element.

## Tests

```sh
swift test
```

The tests check the building blocks, stream words, draws and derived keys against every vector
of the specification (`Fixtures/vectors.json`, a copy of the spec repository's file), and the
CPU fills against tandem-c's stream dumps for u32 at K = 32 and K = 8, u64, f32 and f64, from
starts that cut rows. They check the scalar bounded draws, the bounded fills of tandem-c and
tandem-cuda at starts 0, 1 and 12345, the normal pairs and fills, and the exponentials against
the fixtures of tandem-c and tandem-cuda (`Fixtures/cross.json`, from `tools/gen_fixtures.sh`),
all bit for bit. They check that a bounded fill cut at element boundaries equals the whole fill
at positions with rejections, and that empty fills keep the position rules. They hash 10^7 normals
and 5 10^6 exponentials of each precision as tandem-c's `test_normal_bits.c` and
`test_exponential_bits.c` do, f32 on the GPU, and require tandem-c's hashes. For 10^7 normals and
10^7 exponentials of each precision they check raw moments 1 to 4 within 5 standard errors and a
Kolmogorov-Smirnov distance under the critical value at level 0.001. The fixtures are those of
tandem-c `4e9a69f` and tandem-cuda `c5c5725`.

On the GPU they check every fill kind against the CPU fill of the same kind at K = 32, 8 and 1,
from even and odd first draws, for odd counts and fills that span threadgroups, and at offsets
that misalign the 16-byte stores, with a sentinel that catches stores outside the fill. They
check the GPU fills against tandem-c's dumps and the fixtures above, the cut bounded fill, and
empty fills. The GPU tests skip when there is no Metal device. CI runs `swift build` and
`swift test` on macos-15 and macos-26 runners and fails when the embedded shader drifts from
`tandem.metal`.

## Speed

`swift run -c release tandem-bench` prints the GPU fills into device memory, no readback, 16
fills per command buffer timed on the GPU, best of 7, and the CPU fills on one core, best of 5.
Apple M4 Pro, macOS 26, Swift 6.3.3, GiB/s of output:

| GPU fill | 2^24 | 2^26 |
|---|---|---|
| `.u32` | 215 | 179 |
| `.u64` | 194 | 171 |
| `.f32` | 230 | 190 |
| `.u32Below(range: 1000)` | 227 | 182 |
| `.u32BelowWide(range: 1000)` | 182 | 170 |
| `.u64Below(range: 1000)` | 177 | 172 |
| `.u32Below(range: 2^31 + 1)` | 17.5 | 18.0 |
| `.normalF32` | 214 | 195 |
| `.normalF32`, odd first draw | 159 | 165 |
| `.exponentialF32` | 204 | 197 |

A range of `2^31 + 1` rejects half the draws, and each rejected draw computes a block of its
fallback stream from the key.

| CPU fill, 2^22 | GiB/s |
|---|---|
| `fillU32` | 8.3 |
| `fillU64` | 7.6 |
| `fillF32` | 7.7 |
| `fillF64` | 7.7 |
| `fillU32(_:below: 1000)` | 4.0 |
| `fillNormalF64` | 2.5 |
| `fillNormalF32` | 1.9 |
| `fillExponentialF64` | 3.3 |
| `fillExponentialF32` | 2.4 |

On the same machine tandem-c fills 18.7 GiB/s of u32, 5.0 of f64 normals and 6.1 of f64
exponentials, so the CPU fills run at 35% to 54% of tandem-c. LLVM folds the row transpose into
the shuffles of the step, and Swift has no shuffle intrinsic to keep them apart.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
