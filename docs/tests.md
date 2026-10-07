# Tests

```sh
swift test
```

## Suite

The tests check the building blocks, stream words, draws and derived keys against every vector
of the specification, and the CPU fills against tandem-c's stream dumps for u32 at K = 32 and
K = 8, u64, f32 and f64, from starts that cut rows.

`ConformanceTests.swift` reads the spec's conformance files and checks the items of its
`conformance/CHECKLIST.md` at 2a4bd08. Every bounded, normal, exponential and weighted choice case runs on
the CPU, whole, cut into two pieces, and as scalar draws where a scalar draw is element 0 of a
fill. Every case of a kind Metal draws runs on the GPU, whole and cut. Values and end positions
match bit for bit. The tests also check the scalar bounded draws, the fallback index of rejected
draws and missed normals, the width the fills name, the Float32 normal pairs and odd counts, the
choice tables and rejected weights, and reads across blocks, rows and chunks. They hash the stream
dumps and the long outputs of tandem-c's dump tools, f32 on the GPU, against the SHA-256 and
FNV-1a values of `hashes.json`. A start position at or past 2^63 and a fill that reaches 2^64
stop the process by a precondition, so those items are not run, and the port has no complex
draws.

They check that a bounded fill cut at element boundaries equals the whole fill at positions with
rejections, and that a Float64 normal fill cut at a missed element equals the whole fill and the
scalar draws. For 10^7 normals and 10^7 exponentials of each precision they check raw moments 1
to 4 within 5 standard errors and a Kolmogorov-Smirnov distance under the critical value at level
0.001.

On the GPU they check every fill kind, weighted choice included, against the CPU fill of the same
kind at K = 32, 8 and 1, from even and odd first draws, for odd counts and fills that span
threadgroups, and at offsets that misalign the 16-byte stores, with a sentinel that catches stores
outside the fill. They check the GPU fills against tandem-c's dumps, the cut bounded fill, and
empty fills. The GPU tests skip when there is no Metal device.

## Fixtures

`Fixtures/vectors.json` is a copy of the spec repository's file. `Fixtures/conformance` holds the
spec's `conformance/*.json` at commit `2a4bd08`, and the `.bin` files are tandem-c's stream dumps.
`tools/gen_fixtures.sh` copies them.

## CI

CI runs `swift build` and `swift test` on macos-15 and macos-26 runners. It fails when the
embedded shader drifts from `tandem.metal`, the ziggurat tables from the spec's table file, or the
conformance files from the spec at `2a4bd08`.
