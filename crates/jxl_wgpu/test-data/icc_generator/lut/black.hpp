#pragma once
#include "profile.hpp"

namespace lut {
inline Values Evaluate(const Pipeline &pipeline, Values input, bool native) {
  for (const auto &stage : pipeline.stages)
    input = stage.apply(input, native);
  return input;
}

inline Values DetectBlack(const Profile &profile, unsigned intent,
                          bool native) {
  const auto channels = profile.recipe.channels;
  if (channels != 1 && channels != 3 && channels != 4)
    return Values(3, {0, 0});
  Values input(channels, {channels == 4 ? 1.0 : 0.0, 0});
  auto xyz = Evaluate(profile.pipelines[intent * 2], input, native);
  auto lab = Lab(false).apply(xyz, native);
  const auto clip = [](double lightness) {
    return lightness > 95 ? 0.0 : std::clamp(lightness, 0.0, 50.0);
  };
  const double lo = lab[0].x - lab[0].radius, hi = lab[0].x + lab[0].radius;
  const double center = clip(lab[0].x);
  double radius =
      std::max(std::abs(center - clip(lo)), std::abs(center - clip(hi)));
  // The strict L*>95 reset is discontinuous. Enclose both branches if the
  // independently propagated input interval straddles that threshold.
  if (lo <= 95 && hi > 95)
    radius = std::max({radius, std::abs(center), std::abs(center - 50)});
  lab[0] = {center, radius, lab[0].semantics};
  return Lab(true).apply(lab, native);
}

inline Stage BlackConnection(Values primary_black, Values native_black,
                             std::array<double, 3> target) {
  return {{}, [=](const Values &input, bool native) {
            const auto &black = native ? native_black : primary_black;
            const std::array<double, 3> white{.9642, 1, .8249};
            Values output;
            for (unsigned c = 0; c < 3; ++c) {
              Check(std::abs(white[c] - black[c].x) > black[c].radius,
                    "black interval must exclude singular compensation");
              const auto evaluate = [&](double x, double b) {
                return target[c] +
                       (white[c] - target[c]) * (x - b) / (white[c] - b);
              };
              const double center = evaluate(input[c].x, black[c].x);
              double radius = 0;
              for (double x :
                   {input[c].x - input[c].radius, input[c].x + input[c].radius})
                for (double b : {black[c].x - black[c].radius,
                                 black[c].x + black[c].radius})
                  radius = std::max(radius, std::abs(evaluate(x, b) - center));
              const double scale =
                  (white[c] - target[c]) / (white[c] - black[c].x);
              radius += 16 * epsilon *
                        (1 + std::abs(center) + std::abs(scale * input[c].x) +
                         std::abs(scale * black[c].x));
              output.push_back(
                  {center, radius, input[c].semantics | black[c].semantics});
            }
            return output;
          }};
}

inline std::vector<Profile> BlackProfiles() {
  std::vector<Profile> profiles;
  for (auto format : {Format::Eight, Format::Sixteen})
    for (bool lab : {false, true})
      for (unsigned channels : {1u, 3u, 4u, 5u}) {
        const std::string name =
            std::string(format == Format::Eight ? "lut8_" : "lut16_") +
            (lab ? "lab_" : "xyz_") + std::to_string(channels);
        profiles.push_back(Build({name, format, lab, channels, 2}));
      }
  for (unsigned channels : {1u, 3u}) {
    Recipe low{"low_" + std::to_string(channels), Format::Sixteen, false,
               channels, 2};
    low.output_floor = 0;
    profiles.push_back(Build(low));
    Recipe bright{"bright_" + std::to_string(channels), Format::Sixteen, false,
                  channels, 2};
    bright.output_floor = .75;
    bright.output_gain = .25;
    profiles.push_back(Build(bright));
  }
  return profiles;
}
} // namespace lut
