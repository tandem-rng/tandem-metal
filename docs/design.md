# Design

Bounded integers, normals and exponentials follow Appendix A of the specification. An empty
bounded, Float32 normal or exponential fill leaves the position, and an empty uniform or Float64
normal fill aligns it.

## Bounded integers

A bounded fill maps element `i` to draw `i`. A draw that Lemire rejects retries on `split(g)`
of `purpose(P_w)` of the key, where `g` is the draw's index in the stream, so a fill cut at any
element boundary equals the whole fill. A scalar bounded draw rejects onto the next draws of
the stream instead.

## Normals and exponentials

Float64 normals are the 1024-layer ziggurat of Appendix A, on the CPU. Element `i` takes 64-bit
draw `i`. A draw that misses the inner rectangles, 0.43 % of them, continues on `split(g)` of
`purpose(0x4e524d3634)` of the key, where `g` is the draw's index in the stream, and leaves the
position alone. A fill cut at any element therefore equals the whole fill.
`tools/gen_zig_tables.swift` writes `Sources/Tandem/ZigTables.swift` from the spec's
`tables/normal_f64_zig1024.json` after it checks the file's SHA-256.

Float32 normal pair `j` is elements `2j` and `2j + 1` from uniforms `2j` and `2j + 1`, so an odd
count consumes one uniform more than it writes.

The normals and exponentials copy the arithmetic of tandem-c: a short series for `log` on the
exponent-split argument, and series for `cos` and `sin` on an angle cut at the nearest quarter
turn, with every multiply-add an explicit `fma`. The ziggurat takes `log` only on a missed draw. The shader is compiled with fast math off and
takes division and square root from `precise::`, which round correctly. The GPU f32 normals and
exponentials and the CPU f32 and f64 ones are then bit for bit those of tandem-c.
