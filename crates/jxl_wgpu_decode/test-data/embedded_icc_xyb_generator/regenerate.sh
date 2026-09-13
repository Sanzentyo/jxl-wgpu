#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo 'usage: regenerate.sh OUTPUT_DIRECTORY' >&2
  exit 2
fi
mkdir -p "$1"
output=$(cd "$1" && pwd)
repository=$(CDPATH= cd -- "$(dirname -- "$0")/../../../.." && pwd)
cd "$repository"
generator=crates/jxl_wgpu_decode/test-data/embedded_icc_xyb_generator
profiles=crates/jxl_wgpu_decode/test-data/embedded_icc
work=$(mktemp -d "${TMPDIR:-/tmp}/jxl-icc-xyb.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

for program in linear convert animation compose; do
  c++ -std=c++17 -Wall -Wextra -Werror -ffp-contract=off \
    -Itools/jxl_test_support/native "$generator/$program.cpp" -o "$work/$program" \
    $(pkg-config --cflags --libs libjxl libjxl_cms lcms2)
done
"$work/linear" "$profiles" "$work/linear-data"
"$work/convert" "$work/linear-data" "$profiles" "$work/converted"
"$work/animation" "$profiles" "$work/layers"
"$work/compose" "$work/layers" "$profiles" "$work/composed"

mkdir -p "$output/animation"
for color in rgb gray; do
  for codec in modular vardct; do
    name=${color}_${codec}_xyb
    for mode in builtin_0 builtin_1 builtin_2 cms_0 cms_1 cms_2; do
      cmp "$work/linear-data/${name}_builtin_0.f32le" "$work/linear-data/${name}_${mode}.f32le"
    done
    cp "$work/linear-data/${name}_builtin_0.f32le" "$output/${name}.linear.native.f32le"
    for target in rgb gray; do
      for reference in native scalar; do
        cp "$work/converted/${name}_to_${target}.${reference}.f32le" "$output/${name}.${target}.${reference}.f32le"
      done
    done
    for frame in 0 1; do
      cmp "$work/layers/${color}_${codec}_layers_builtin.frame${frame}.linear.f32le" \
        "$work/layers/${color}_${codec}_layers_cms.frame${frame}.linear.f32le"
    done
    cp "$work/layers/${color}_${codec}.jxl" "$output/animation/"
  done
done
cp "$work/composed/"*.f32le "$output/animation/"
