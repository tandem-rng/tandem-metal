# Tests

```sh
swift test
```

## Suite

The tests check the building blocks, stream words, draws and derived keys against every vector
of the specification, and the CPU fills against tandem-c's stream dumps for u32 at K = 32 and
K = 8, u64, f32 and f64, from starts that cut rows. They check the scalar bounded draws, the
bounded fills of tandem-c and tandem-cuda at starts 0, 1 and 12345, the ziggurat normals as
fills and scalar draws through every path, the normal pairs and fills, and the exponentials
against the fixtures of tandem-c and tandem-cuda, all bit for bit. They check that a bounded
fill cut at element boundaries equals the whole fill at positions with rejections, that a
Float64 normal fill cut at a missed element equals the whole fill and the scalar draws, and
that empty fills keep the position rules. As tandem-c's `test_normal_bits.c` and
`test_exponential_bits.c` do, they hash fills from seed (2026, 7) at starts 0, 1, 77, 12345 and
2^30, f32 on the GPU, and require tandem-c's FNV-1a hashes:

| Fill per start | FNV-1a |
|---|---|
| 10^6 Float64 normals | `a61cfa844c85f7c1` |
| 2 10^6 - 1 Float32 normals | `aa1ea656ce73a4fb` |
| 10^6 exponentials of each precision | `47f8f98297d94ee2` |

For 10^7 normals
and 10^7 exponentials of each precision they check raw moments 1 to 4 within 5 standard errors
and a Kolmogorov-Smirnov distance under the critical value at level 0.001.

On the GPU they check every fill kind against the CPU fill of the same kind at K = 32, 8 and 1,
from even and odd first draws, for odd counts and fills that span threadgroups, and at offsets
that misalign the 16-byte stores, with a sentinel that catches stores outside the fill. They
check the GPU fills against tandem-c's dumps and the fixtures above, the cut bounded fill, and
empty fills. The GPU tests skip when there is no Metal device.

## Fixtures

`Fixtures/vectors.json` is a copy of the spec repository's file. `Fixtures/cross.json` holds
the fixtures of tandem-c and tandem-cuda, from `tools/gen_fixtures.sh`. The fixtures are those
of tandem-c `121db59` and tandem-cuda `0ff5f18`. tandem-c's `tests/cross_normal.h` has SHA-256
`3cd7c8f9178711255718288eb712eaccb33a1726d2a185f412f13590398ad3ac`, and tandem-cuda's
`tests/cross_fill_normal.h` `d7057fdcd44b9c961c66fb7580520c1c5ec4665ef311a4b7c8248cd36e6c0cee`.

## CI

CI runs `swift build` and `swift test` on macos-15 and macos-26 runners. It fails when the
embedded shader drifts from `tandem.metal` or the ziggurat tables from the spec's table file.
