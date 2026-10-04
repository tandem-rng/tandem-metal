# Speed

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
