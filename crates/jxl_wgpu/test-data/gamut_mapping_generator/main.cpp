// Offline reference: call the pinned libjxl scalar primitive directly.
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>
#include "lib/jxl/cms/tone_mapping.h"

int main(int argc, char** argv) {
  if (argc != 2) throw std::runtime_error("expected output path");
  std::ofstream out(argv[1]);
  out << std::setprecision(9);
  const std::array<jxl::Vector3, 3> weights = {{
    {0.212639005871510f, 0.715168678767756f, 0.072192315360734f},
    {0.262700212011267f, 0.677998071518871f, 0.059301716469862f},
    {0.228974564069749f, 0.691738521836506f, 0.079286914093745f},
  }};
  std::vector<jxl::Color> colors;
  constexpr std::array<float, 7> values = {-.25f, 0.f, .125f, .5f, 1.f, 1.25f, 4.f};
  for (float r : values) for (float g : values) for (float b : values) colors.push_back({r,g,b});
  for (float edge : {0.f, 1.f}) {
    for (float value : {std::nextafter(edge, -1.f), edge, std::nextafter(edge, 2.f)}) {
      colors.push_back({value, .5f, .5f});
      colors.push_back({.5f, value, .5f});
      colors.push_back({.5f, .5f, value});
    }
  }
  for (size_t space = 0; space < weights.size(); ++space) {
    for (float preference : {0.f, .1f, .5f, .9f, 1.f}) {
      for (const auto& color : colors) {
        const auto& w = weights[space];
        if (color[0] * w[0] + color[1] * w[1] + color[2] * w[2] < 0.f) continue;
        auto mapped = color;
        jxl::GamutMapScalar(mapped, w, preference);
        out << space << ' ' << preference;
        for (float value : color) out << ' ' << value;
        for (float value : mapped) {
          if (!std::isfinite(value)) throw std::runtime_error("nonfinite native output");
          out << ' ' << value;
        }
        out << '\n';
      }
    }
  }
  if (!out) throw std::runtime_error("reference output failed");
}
