# Design

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

Element `i` of a fill is draw `i`, so threads, command buffers or devices that
start at the position of their first element, or draw from `split(task)`, reproduce a serial
run for any decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows. Start every range of a normal fill at an even element.
