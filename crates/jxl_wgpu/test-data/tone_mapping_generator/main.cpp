// Offline oracle: invoke the pinned libjxl primitive, without copying its equations.
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
  for (float white : {100.f, 1000.f, 4000.f, 10000.f}) {
    for (float peak : {80.f, 255.f, 1000.f}) {
      if (peak >= white) continue;
      for (float black : {0.f, 1.f}) {
        const float display_black = black == 0 ? 0.f : 0.01f;
        const jxl::Rec2408ToneMapperBase mapper({black, white}, {display_black, peak}, {0, 1, 0});
        std::vector<jxl::Color> colors;
        for (int i = 0; i <= 256; ++i) {
          const float y = static_cast<float>(i) / 256;
          colors.push_back({0.7f * y, y, 1.3f * y});
        }
        for (float y : {0.9f * black / white, 1.1f * black / white, 0.f, 0.5e-6f / white,
                        1e-6f / white, 1.5e-6f / white, 0.00001f, 0.001f, 1.1f, 2.f}) {
          colors.push_back({-0.2f * y, y, 1.8f * y});
          colors.push_back({y, y, y});
        }
        for (const auto& color : colors) {
          auto mapped = color;
          mapper.ToneMap(mapped);
          // The primitive's neutral is RGB (1,1,1). Re-express only its black cap as
          // D50 XYZ, the identity MPE profile's device domain used by the resident test.
          if (white * color[1] <= 1e-6f) {
            mapped[0] *= 0.9642f;
            mapped[2] *= 0.8249f;
          }
          out << black << ' ' << white << ' ' << display_black << ' ' << peak;
          for (float v : color) out << ' ' << v;
          for (float v : mapped) {
            if (!std::isfinite(v)) throw std::runtime_error("nonfinite native reference");
            out << ' ' << v;
          }
          out << '\n';
        }
      }
    }
  }
  if (!out) throw std::runtime_error("reference output failed");
}
