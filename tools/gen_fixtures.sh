#!/bin/sh
# Rebuilds Tests/TandemTests/Fixtures from checkouts of the spec and tandem-c:
#   tools/gen_fixtures.sh <tandem-spec> <tandem-c> [spec commit]
# The conformance files come from the spec commit, 2a4bd08 by default, which CI checks.
set -e
spec=$1 c=$2 rev=${3:-2a4bd08}
root=$(cd "$(dirname "$0")/.." && pwd)
out=$root/Tests/TandemTests/Fixtures
mkdir -p "$out/conformance"
for f in below fill_below normal exponential choice hashes; do
    git -C "$spec" show "$rev:conformance/$f.json" > "$out/conformance/$f.json"
done
cp "$spec/vectors.json" "$out/"
for f in k1234_K32_u32 k1234_K8_u32 k1234_K32_u64 seed42_K32_f32 seed42_K32_f64; do
    cp "$c/tests/data/$f.bin" "$out/"
done
