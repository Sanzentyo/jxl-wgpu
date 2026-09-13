#include "cases.h"

#include <utility>

#include "lib/jxl/modular/options.h"

namespace fixtures {
namespace {
jxl::Transform Rct(uint32_t type, uint32_t begin = 0) {
  jxl::Transform transform(jxl::TransformId::kRCT);
  transform.rct_type = type;
  transform.begin_c = begin;
  return transform;
}

jxl::Transform Palette(uint32_t begin, uint32_t count) {
  jxl::Transform transform(jxl::TransformId::kPalette);
  transform.begin_c = begin;
  transform.num_c = count;
  transform.nb_colors = 65535;
  transform.nb_deltas = 0;
  transform.predictor = jxl::Predictor::Zero;
  return transform;
}

jxl::SqueezeParams Split(bool horizontal, bool in_place, uint32_t begin, uint32_t count) {
  jxl::SqueezeParams parameters;
  parameters.horizontal = horizontal;
  parameters.in_place = in_place;
  parameters.begin_c = begin;
  parameters.num_c = count;
  return parameters;
}

jxl::Transform Squeeze(std::vector<jxl::SqueezeParams> parameters = {}) {
  jxl::Transform transform(jxl::TransformId::kSqueeze);
  transform.squeezes = std::move(parameters);
  return transform;
}

void TransformCases(std::vector<Case>& cases) {
  for (uint32_t type = 0; type < 42; ++type) {
    Case test;
    test.name = "rct_" + std::to_string(type);
    test.selectors = {0, 0, 0};
    test.global_transforms = {Rct(type)};
    cases.push_back(test);
  }
  for (int cb = 0; cb < 4; ++cb) for (int y = 0; y < 4; ++y) for (int cr = 0; cr < 4; ++cr) {
    Case test;
    test.name = "squeeze_sampling_" + std::to_string(cb) + std::to_string(y) + std::to_string(cr);
    test.selectors = {cb, y, cr};
    test.global_transforms = {Squeeze()};
    cases.push_back(test);
  }
  for (auto selectors : {std::array<int, 3>{0, 0, 0}, {0, 1, 0}, {1, 2, 3}}) {
    for (auto size : {std::array<size_t, 2>{1, 1}, {1, 19}, {37, 1}}) {
      Case test;
      test.name = "squeeze_thin_" + std::to_string(selectors[0]) + std::to_string(selectors[1]) + std::to_string(selectors[2])
          + "_" + std::to_string(size[0]) + "x" + std::to_string(size[1]);
      test.selectors = selectors;
      test.width = size[0]; test.height = size[1];
      test.global_transforms = {Squeeze()};
      cases.push_back(test);
    }
  }
  for (uint32_t shift = 0; shift <= 3; ++shift) for (bool vertical : {false, true}) {
    Case test;
    test.name = "squeeze_groups_" + std::to_string(128u << shift) + (vertical ? "_vertical" : "_horizontal");
    test.group_size_shift = shift;
    if (vertical) test.height = (256u << shift) + 3; else test.width = (256u << shift) + 3;
    test.global_transforms = {Squeeze()};
    cases.push_back(test);
  }
  for (bool in_place : {false, true}) {
    Case test;
    test.name = in_place ? "squeeze_in_place" : "squeeze_append";
    test.selectors = {1, 2, 3};
    test.width = 259; test.height = 129; test.group_size_shift = 0;
    test.global_transforms = {Squeeze({Split(true, in_place, 0, 3), Split(false, in_place, 0, 3)})};
    cases.push_back(test);
  }
  Case lf;
  lf.name = "squeeze_lf"; lf.width = 2051; lf.group_size_shift = 0; lf.passes = 2;
  std::vector<jxl::SqueezeParams> splits;
  for (size_t level = 0; level < 3; ++level) {
    splits.push_back(Split(true, true, 0, 3));
    splits.push_back(Split(false, true, 0, 3));
  }
  lf.global_transforms = {Squeeze(splits)};
  cases.push_back(lf);
  Case passes;
  passes.name = "squeeze_passes"; passes.width = 259; passes.height = 37;
  passes.group_size_shift = 0; passes.passes = 2; passes.global_transforms = {Squeeze()};
  cases.push_back(passes);
  for (uint32_t bits : {8, 31}) {
    Case test;
    test.name = "squeeze_integer_" + std::to_string(bits); test.bits = bits;
    test.global_transforms = {Squeeze()}; cases.push_back(test);
  }
  for (uint32_t bits : {16, 24, 32}) {
    Case test;
    test.name = "squeeze_float_" + std::to_string(bits); test.bits = bits;
    test.exponent_bits = bits == 16 ? 5 : bits == 24 ? 7 : 8;
    test.positive_samples = bits == 32;
    test.global_transforms = {Squeeze()}; cases.push_back(test);
  }
  for (uint32_t channel = 0; channel < 3; ++channel) for (bool grouped : {false, true}) {
    Case test;
    test.name = std::string(grouped ? "palette_groups_" : "palette_") + std::to_string(channel);
    test.selectors = {1, 2, 3};
    if (grouped) { test.width = 259; test.height = 37; test.group_size_shift = 0; }
    test.global_transforms = {Palette(channel, 1)}; cases.push_back(test);
  }
  Case palette;
  palette.name = "palette_rgb"; palette.selectors = {0, 0, 0};
  palette.global_transforms = {Palette(0, 3)}; cases.push_back(palette);
  palette.name = "rct_palette_squeeze";
  palette.global_transforms = {Rct(6), Palette(0, 3), Squeeze()}; cases.push_back(palette);
  palette.name = "palette_squeeze"; palette.selectors = {1, 2, 3};
  palette.global_transforms = {Palette(0, 1), Squeeze()}; cases.push_back(palette);
  Case residual;
  residual.name = "squeeze_residual_rct"; residual.selectors = {0, 0, 0};
  residual.global_transforms = {Squeeze({Split(true, false, 0, 3)}), Rct(41, 3)};
  cases.push_back(residual);
  Case extras;
  extras.name = "squeeze_extras"; extras.width = 259; extras.height = 37; extras.group_size_shift = 0;
  extras.extra_factors = {2, 8};
  extras.global_transforms = {Squeeze({Split(true, false, 0, 3), Split(false, false, 0, 3), Split(true, false, 3, 1)})};
  cases.push_back(extras);
  Case restored;
  restored.name = "restored_palette_squeeze"; restored.selectors = {1, 2, 3};
  restored.gaborish = true; restored.epf = 3;
  restored.global_transforms = {Palette(0, 1), Squeeze()}; cases.push_back(restored);
  for (uint32_t type = 0; type < 42; ++type) for (bool horizontal : {true, false}) {
    Case test;
    test.name = "rct_empty_" + std::to_string(type) + (horizontal ? "_1x19" : "_37x1");
    test.selectors = {0, 0, 0};
    if (horizontal) test.width = 1; else test.height = 1;
    test.global_transforms = {Squeeze({Split(horizontal, false, 0, 3)}), Rct(type, 3)};
    cases.push_back(test);
  }
}

Case Grouped(std::string name) {
  Case test;
  test.name = std::move(name);
  test.width = 259; test.height = 37; test.group_size_shift = 0;
  return test;
}

void LocalCases(std::vector<Case>& cases) {
  for (uint32_t type = 0; type < 42; ++type) {
    Case test = Grouped("local_rct_" + std::to_string(type));
    test.height = 129; test.selectors = {0, 0, 0};
    test.pass_transforms = {Rct(type)}; cases.push_back(test);
  }
  for (int cb = 0; cb < 4; ++cb) for (int y = 0; y < 4; ++y) for (int cr = 0; cr < 4; ++cr) {
    Case test = Grouped("local_squeeze_sampling_" + std::to_string(cb) + std::to_string(y) + std::to_string(cr));
    test.selectors = {cb, y, cr}; test.pass_transforms = {Squeeze()}; cases.push_back(test);
  }
  for (uint32_t channel = 0; channel < 3; ++channel) {
    Case test = Grouped("local_palette_" + std::to_string(channel));
    test.selectors = {1, 2, 3}; test.pass_transforms = {Palette(channel, 1)}; cases.push_back(test);
  }
  Case palette = Grouped("local_palette_rgb");
  palette.selectors = {0, 0, 0}; palette.pass_transforms = {Palette(0, 3)}; cases.push_back(palette);
  palette.name = "local_rct_palette_squeeze"; palette.width = 257; palette.height = 129;
  palette.pass_transforms = {Rct(6), Palette(0, 3), Squeeze()}; cases.push_back(palette);
  Case residual = Grouped("local_squeeze_residual_rct");
  residual.width = 257; residual.height = 129; residual.selectors = {0, 0, 0};
  residual.pass_transforms = {Squeeze({Split(true, false, 0, 3)}), Rct(41, 3)}; cases.push_back(residual);
  for (bool in_place : {false, true}) {
    Case test = Grouped(in_place ? "local_squeeze_in_place" : "local_squeeze_append");
    test.height = 129; test.selectors = {1, 2, 3};
    test.pass_transforms = {Squeeze({Split(true, in_place, 0, 3), Split(false, in_place, 0, 3)})};
    cases.push_back(test);
  }
  for (uint32_t shift = 0; shift <= 3; ++shift) for (bool vertical : {false, true}) {
    Case test;
    test.name = "local_squeeze_groups_" + std::to_string(128u << shift) + (vertical ? "_vertical" : "_horizontal");
    test.group_size_shift = shift;
    if (vertical) test.height = (256u << shift) + 1; else test.width = (256u << shift) + 1;
    test.pass_transforms = {Squeeze()}; cases.push_back(test);
  }
  Case extras = Grouped("local_squeeze_extras");
  extras.extra_factors = {2, 8};
  extras.pass_transforms = {Squeeze({Split(true, false, 0, 3), Split(false, false, 0, 3)})};
  cases.push_back(extras);
  for (int kind = 0; kind < 3; ++kind) {
    Case test;
    test.name = kind == 0 ? "local_lf_squeeze" : kind == 1 ? "local_lf_palette" : "local_lf_rct";
    test.width = 2051; test.group_size_shift = 0; test.selectors = {0, 0, 0}; test.passes = 2;
    std::vector<jxl::SqueezeParams> splits;
    for (size_t level = 0; level < 3; ++level) {
      splits.push_back(Split(true, true, 0, 3));
      splits.push_back(Split(false, true, 0, 3));
    }
    test.global_transforms = {Squeeze(splits)};
    test.lf_transforms = kind == 0 ? std::vector<jxl::Transform>{Squeeze()}
        : kind == 1 ? std::vector<jxl::Transform>{Palette(0, 3), Squeeze()}
                    : std::vector<jxl::Transform>{Rct(41)};
    test.pass_transforms = {Squeeze()}; cases.push_back(test);
  }
  Case passes = Grouped("local_passes");
  passes.passes = 2; passes.global_transforms = {Squeeze()}; passes.pass_transforms = {Squeeze()};
  cases.push_back(passes);
  Case restored = Grouped("local_restored_palette_squeeze");
  restored.selectors = {1, 2, 3}; restored.gaborish = true; restored.epf = 3;
  restored.pass_transforms = {Palette(0, 1), Squeeze()}; cases.push_back(restored);
  for (uint32_t bits : {8, 31}) {
    Case test = Grouped("local_squeeze_integer_" + std::to_string(bits));
    test.bits = bits; test.pass_transforms = {Squeeze()}; cases.push_back(test);
  }
  for (uint32_t bits : {16, 24, 32}) {
    Case test = Grouped("local_squeeze_float_" + std::to_string(bits));
    test.bits = bits; test.exponent_bits = bits == 16 ? 5 : bits == 24 ? 7 : 8;
    test.positive_samples = bits == 32; test.pass_transforms = {Squeeze()}; cases.push_back(test);
  }
  for (uint32_t factor : {2, 4, 8}) {
    Case test = Grouped("local_resampling_" + std::to_string(factor));
    test.width = 256 * factor + 3; test.bits = 32; test.exponent_bits = 8; test.positive_samples = true;
    test.upsampling = factor; test.extra_factors = {factor, 8};
    test.pass_transforms = {Squeeze({Split(true, false, 0, 3), Split(false, false, 0, 3)})};
    cases.push_back(test);
  }
}
void FeatureSources(std::vector<Case>& cases) {
  for (uint32_t factor : {1, 2, 4, 8}) for (bool local : {false, true}) {
    Case test = Grouped(std::string(local ? "feature_local_up" : "feature_global_up") + std::to_string(factor));
    test.width = 256 * factor + 3;
    test.bits = 32; test.exponent_bits = 8; test.positive_samples = true;
    test.upsampling = factor; test.extra_factors = {factor, factor};
    test.selectors = local ? std::array<int, 3>{1, 2, 3} : std::array<int, 3>{0, 1, 0};
    auto& transforms = local ? test.pass_transforms : test.global_transforms;
    transforms = {Squeeze({Split(true, false, 0, 3), Split(false, false, 0, 3)})};
    cases.push_back(test);
  }
  for (auto selectors : {std::array<int, 3>{0, 1, 0}, {1, 2, 3}}) {
    Case test;
    test.name = "feature_mixed_" + std::to_string(selectors[0]) + std::to_string(selectors[1]) + std::to_string(selectors[2]);
    test.width = 257; test.height = 17; test.bits = 8;
    test.selectors = selectors;
    cases.push_back(test);
  }
}
}  // namespace

std::vector<Case> Cases() {
  std::vector<Case> cases;
  for (int cb = 0; cb < 4; ++cb) {
    for (int y = 0; y < 4; ++y) {
      for (int cr = 0; cr < 4; ++cr) {
        Case test;
        test.name = "sampling_" + std::to_string(cb) + std::to_string(y) + std::to_string(cr);
        test.selectors = {cb, y, cr};
        cases.push_back(test);
      }
    }
  }
  for (uint32_t bits : {8, 12, 31}) {
    Case test;
    test.name = "integer_" + std::to_string(bits);
    test.bits = bits;
    cases.push_back(test);
  }
  for (uint32_t bits : {16, 24, 32}) {
    Case test;
    test.name = "float_" + std::to_string(bits);
    test.bits = bits;
    test.exponent_bits = bits == 16 ? 5 : bits == 24 ? 7 : 8;
    cases.push_back(test);
  }
  for (bool vertical : {false, true}) {
    Case test;
    test.name = vertical ? "thin_vertical" : "thin_horizontal";
    if (vertical) test.width = 1; else test.height = 1;
    cases.push_back(test);
  }
  Case gray;
  gray.name = "gray";
  gray.gray = true;
  cases.push_back(gray);
  for (uint32_t epf : {1, 2, 3}) {
    Case test;
    test.name = "restoration_" + std::to_string(epf);
    test.selectors = {1, 2, 3};
    test.gaborish = true;
    test.epf = epf;
    cases.push_back(test);
  }
  for (uint32_t factor : {2, 4, 8}) {
    Case test;
    test.name = "resampling_" + std::to_string(factor);
    test.width = 53;
    test.height = 35;
    test.bits = 32;
    test.exponent_bits = 8;
    test.upsampling = factor;
    test.extra_factors = {factor, 8};
    test.gaborish = true;
    test.epf = 2;
    cases.push_back(test);
  }
  Case associated;
  associated.name = "associated";
  associated.bits = 32;
  associated.exponent_bits = 8;
  associated.associated = true;
  associated.extra_factors = {1, 1};
  cases.push_back(associated);
  for (uint32_t orientation = 2; orientation <= 8; ++orientation) {
    Case test = associated;
    test.name = "orientation_" + std::to_string(orientation);
    test.orientation = orientation;
    cases.push_back(test);
  }
  for (uint32_t shift = 0; shift <= 3; ++shift) {
    for (bool vertical : {false, true}) {
      Case test;
      test.name = "groups_" + std::to_string(128u << shift) + (vertical ? "_vertical" : "_horizontal");
      test.group_size_shift = shift;
      if (vertical) test.height = (256u << shift) + 3; else test.width = (256u << shift) + 3;
      cases.push_back(test);
    }
  }
  Case prefix;
  prefix.name = "global_prefix";
  prefix.width = 257;
  cases.push_back(prefix);
  prefix.name = "passes_global_prefix";
  prefix.passes = 2;
  cases.push_back(prefix);
  for (bool directional : {false, true}) {
    Case test;
    test.name = directional ? "passes_directional" : "passes_420";
    test.width = 259;
    test.group_size_shift = 0;
    test.passes = 2;
    if (directional) test.selectors = {1, 2, 3};
    cases.push_back(test);
  }
  Case lf;
  lf.name = "lf_extras";
  lf.width = 2051;
  lf.group_size_shift = 0;
  lf.bits = 32;
  lf.exponent_bits = 8;
  lf.extra_factors = {8, 8};
  cases.push_back(lf);
  TransformCases(cases);
  LocalCases(cases);
  FeatureSources(cases);
  return cases;
}

}  // namespace fixtures
