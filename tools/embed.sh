#!/bin/sh
# Embeds tandem.metal in the library as a string, so the package needs no resource bundle.
set -e
cd "$(dirname "$0")/.."
{
    printf '// Generated from tandem.metal by tools/embed.sh. Do not edit.\n\nlet shaderSource = #"""\n'
    cat tandem.metal
    printf '"""#\n'
} > Sources/Tandem/Shader.swift
