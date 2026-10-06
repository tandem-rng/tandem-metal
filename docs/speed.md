# Speed

`swift run -c release tandem-bench` prints the GPU fills into device memory, no readback, 16
fills per command buffer timed on the GPU, best of 7, and the CPU fills on one core, best of 5.
Apple M4 Pro, macOS 26, Swift 6.3.3, GiB/s of output, every figure from one session.

## CPU

The baselines are the fastest generators Swift reaches: `arc4random_buf` for raw words, and a
loop over libc's `drand48` for the rest, with Box-Muller pairs for the normals and
`-log(1 - u)` for the exponentials. `drand48` runs four times or more faster than
`SystemRandomNumberGenerator` and GameplayKit's sources.

| CPU fill, 2^22 | Tandem | baseline |
|---|---|---|
| `fillU32` | 8.89 | 4.49 |
| `fillU64` | 8.78 | 4.44 |
| `fillF32` | 7.77 | 2.96 |
| `fillF64` | 7.79 | 5.94 |
| `fillU32(_:below: 1000)` | 3.54 | 2.76 |
| `fillNormalF64` | 3.66 | 1.02 |
| `fillNormalF32` | 1.85 | 0.50 |
| `fillExponentialF64` | 3.19 | 1.38 |
| `fillExponentialF32` | 2.38 | 1.49 |

On the same machine tandem-c fills 18.7 GiB/s of u32, 7.1 of f64 normals and 6.1 of f64
exponentials, so the CPU fills run at 35% to 55% of tandem-c. LLVM folds the row transpose into
the shuffles of the step, and Swift has no shuffle intrinsic to keep them apart.

## GPU

The baseline is MPSMatrixRandomPhilox, Philox4x32-10 from Metal Performance Shaders, timed the
same way into the same buffer. MPS has uint32 words, float32 uniforms and float32 normals. The
rows it has no draw for compare with its uint32 words of the same byte count, or its float32
uniforms for the exponentials, marked *.

| GPU fill | 2^24 | MPS Philox | 2^26 | MPS Philox |
|---|---|---|---|---|
| `.u32` | 226 | 174 | 190 | 171 |
| `.u64` | 211 | 175* | 182 | 159* |
| `.f32` | 236 | 170 | 189 | 171 |
| `.u32Below(range: 1000)` | 224 | 173* | 191 | 172* |
| `.u32BelowWide(range: 1000)` | 211 | 175* | 181 | 159* |
| `.u64Below(range: 1000)` | 207 | 175* | 180 | 156* |
| `.u32Below(range: 2^31 + 1)` | 17.5 | 174* | 18.0 | 173* |
| `.normalF32` | 261 | 109 | 189 | 110 |
| `.normalF32`, odd first draw | 159 | 109 | 165 | 110 |
| `.exponentialF32` | 252 | 171* | 151 | 147* |

A range of `2^31 + 1` rejects half the draws, and each rejected draw computes a block of its
fallback stream from the key.
