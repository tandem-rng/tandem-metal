#!/bin/sh
# Rebuilds Tests/TandemTests/Fixtures from checkouts of the spec, tandem-c and tandem-cuda:
#   tools/gen_fixtures.sh <tandem-spec> <tandem-c> <tandem-cuda>
set -e
spec=$1 c=$2 cuda=$3
root=$(cd "$(dirname "$0")/.." && pwd)
out=$root/Tests/TandemTests/Fixtures
tmp=$(mktemp -d)
mkdir "$tmp/cuda"
cp "$c"/tests/cross_*.h "$tmp/"
cp "$cuda"/tests/cross_fill_*.h "$tmp/cuda/"
cc -std=c11 -I "$tmp" "$root/tools/gen_fixtures.c" -o "$tmp/gen"
"$tmp/gen" > "$out/cross.json"
rm -r "$tmp"
cp "$spec/vectors.json" "$out/"
for f in k1234_K32_u32 k1234_K8_u32 k1234_K32_u64 seed42_K32_f32 seed42_K32_f64; do
    cp "$c/tests/data/$f.bin" "$out/"
done
