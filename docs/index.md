# tandem-metal

Metal and Swift implementation of Tandem8x32. Metal kernels fill GPU buffers and a Swift package
draws on the CPU, bit for bit with the
[specification](https://github.com/tandem-rng/spec/blob/main/SPEC.md) and tandem-c.

- [API](api.md): the shader, the Swift `Tandem` type, GPU and CPU fills and draws.
- [Design](design.md): the Appendix A rules and the arithmetic shared with tandem-c.
- [Tests](tests.md): what the suite checks, the fixtures, and what CI runs.
- [Speed](speed.md): Apple M4 Pro figures.

## Install

Add the package to `Package.swift`. It needs swift-tools-version 6.0 and macOS 15.

```swift
// Package.swift
.package(url: "https://github.com/tandem-rng/tandem-metal", branch: "main")
```

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.
