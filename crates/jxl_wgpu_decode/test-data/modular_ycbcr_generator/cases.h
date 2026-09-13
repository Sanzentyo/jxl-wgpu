#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lib/jxl/modular/transform/transform.h"

namespace fixtures {
struct Case {
  std::string name;
  std::array<int, 3> selectors = {0, 1, 0};
  std::size_t width = 37, height = 19;
  uint32_t bits = 16, exponent_bits = 0;
  bool gray = false, gaborish = false, associated = false;
  uint32_t epf = 0, upsampling = 1, orientation = 1;
  uint32_t group_size_shift = 1, passes = 1;
  std::vector<uint32_t> extra_factors;
  std::vector<jxl::Transform> global_transforms;
  std::vector<jxl::Transform> lf_transforms, pass_transforms;
  bool positive_samples = false;
};

std::vector<Case> Cases();
}  // namespace fixtures
