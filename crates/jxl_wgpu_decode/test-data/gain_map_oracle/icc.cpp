// Native and independent profile processing of caller-supplied PCS intervals.
#include "icc.hpp"
#include <icc/rgb.hpp>

int IccOracle(const char *root, const char *name, unsigned intent,
              const char *input, const char *output) try {
  using namespace rgb_icc;
  Check(cmsGetEncodedCMMversion() == 2190, "requires Little CMS 2.19");
  Check(intent < 4, "ICC intent");
  std::vector<TargetSpec> specs(kTargets.begin(), kTargets.end());
  specs.insert(specs.end(), {
      {"lut/lut16_channels2", Recipe{"lut16_channels2", Format::Sixteen, false, 2}},
      {"lut/ab_lab_4", Recipe{"ab_lab_4", Format::AB, true, 4}},
      {"lut/ab_channels5", Recipe{"ab_channels5", Format::AB, true, 5}},
      {"lut/ab_channels15", Recipe{"ab_channels15", Format::AB, false, 15}},
  });
  const auto spec = std::find_if(specs.begin(), specs.end(),
                               [&](const auto &spec) { return std::string(name) == spec.name; });
  Check(spec != specs.end(), "declared ICC target recipe");
  Target target(root, *spec);
  const auto pixels = Sources(input);
  const auto native = target.Native(pixels, intent);
  Bytes records;
  for (size_t p = 0; p < pixels.size(); ++p) {
    const auto exact = target.Evaluate(pixels[p], intent, false);
    auto center = pixels[p];
    for (auto &value : center) value.radius = 0;
    const auto bound = target.Evaluate(center, intent, true);
    for (unsigned c = 0; c < target.channels; ++c)
      Record(records, native[p * target.channels + c], exact[c], bound[c]);
  }
  Save(output, records);
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
