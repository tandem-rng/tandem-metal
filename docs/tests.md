# Tests

```sh
swift test
```

## Suite

The tests check the building blocks, stream words, draws and derived keys against every vector
of the specification, and the CPU fills against tandem-c's stream dumps for u32 at K = 32 and
K = 8, u64, f32 and f64, from starts that cut rows. They check the scalar bounded draws, the
bounded fills of tandem-c and tandem-cuda at starts 0, 1 and 12345, the normal pairs and fills,
and the exponentials against the fixtures of tandem-c and tandem-cuda, all bit for bit. They
check that a bounded fill cut at element boundaries equals the whole fill at positions with
rejections, and that empty fills keep the position rules. They hash 10^7 normals and 5 10^6
exponentials of each precision as tandem-c's `test_normal_bits.c` and
`test_exponential_bits.c` do, f32 on the GPU, and require tandem-c's hashes. For 10^7 normals
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
of tandem-c `4e9a69f` and tandem-cuda `c5c5725`.

## CI

CI runs `swift build` and `swift test` on macos-15 and macos-26 runners and fails when the
embedded shader drifts from `tandem.metal`.
