#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
  echo 'usage: regenerate-alpha.sh LIBJXL_SOURCE OUTPUT_DIRECTORY' >&2
  exit 2
fi
native=$(cd "$1" && pwd)
if [ "$(git -C "$native" rev-parse HEAD)" != a7a9c787341cf703dede03c2009fa460cae5e5df ]; then
  echo 'libjxl v0.12.0 source checkout required' >&2
  exit 2
fi
git -C "$native" diff --exit-code HEAD -- lib/jxl lib/include
mkdir -p "$2"
output=$(cd "$2" && pwd)
repository=$(CDPATH= cd -- "$(dirname -- "$0")/../../../.." && pwd)
cd "$repository"
generator=crates/jxl_wgpu_decode/test-data/embedded_icc_xyb_generator
profiles=crates/jxl_wgpu_decode/test-data/embedded_icc
work=$(mktemp -d "${TMPDIR:-/tmp}/jxl-icc-xyb-alpha.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

c++ -std=c++17 -Wall -Wextra -Werror -ffp-contract=off \
  "$generator/alpha.cpp" -o "$work/alpha" \
  $(pkg-config --cflags --libs libjxl libjxl_cms)
c++ -std=c++17 -Wall -Wextra -Werror -ffp-contract=off \
  -isystem "$native" -Itools/jxl_test_support/native -I"$generator" \
  "$generator/alpha_compose.cpp" "$native/lib/jxl/alpha.cc" -o "$work/compose" \
  $(pkg-config --cflags --libs libjxl libhwy lcms2)
"$work/alpha" "$profiles" "$work/layers"
"$work/compose" "$work/layers" "$profiles" "$work/references"

for color in rgb gray; do
  for codec in modular vardct; do
    for association in straight associated; do
      for mode in 0 1 2 3 4; do
        for suffix in '' '_alpha_ref1'; do
          if [ -n "$suffix" ] && [ "$mode" -ne 2 ]; then continue; fi
          name=${color}_${codec}_${association}_m${mode}${suffix}
        for frame in 0 1; do
          cmp "$work/layers/${name}_layers_builtin.frame${frame}.linear.f32le" \
            "$work/layers/${name}_layers_cms.frame${frame}.linear.f32le"
        done
        cp "$work/layers/${name}.jxl" "$output/"
        done
      done
    done
  done
done
cp "$work/references/"*.f32le "$output/"
