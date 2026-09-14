// Independent v2 LUT source-black policy and Little CMS 2.19 conversions.
#include "lut/black.hpp"
#include <icc/linear.hpp>

namespace {
using namespace lut;

std::vector<float> Native(cmsHPROFILE profile, cmsHPROFILE identity,
                          unsigned channels, int linear,
                          const std::vector<float> &input, unsigned intent) {
  auto device = input;
  if (channels >= 4)
    for (auto &value : device)
      value *= 100;
  auto xyz = linear >= 0 ? cmsCreateXYZProfile() : nullptr;
  const auto format = cmsFormatterForColorspaceOfProfile(profile, 4, TRUE);
  auto transform = cmsCreateTransform(profile, format, xyz ? xyz : identity,
                                      xyz ? TYPE_XYZ_DBL : TYPE_RGB_FLT, intent,
                                      cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE);
  Check(transform, "create native black-point connection");
  const auto pixels = static_cast<cmsUInt32Number>(input.size() / channels);
  std::vector<float> result(pixels * 3);
  if (xyz) {
    std::vector<double> values(pixels * 3);
    cmsDoTransform(transform, device.data(), values.data(), pixels);
    const auto matrix = scalar::Invert(connection::kSpaces[linear].ToPcs());
    for (unsigned pixel = 0; pixel < pixels; ++pixel) {
      const auto rgb =
          connection::Apply(matrix, {values[pixel * 3], values[pixel * 3 + 1],
                                     values[pixel * 3 + 2]});
      for (unsigned c = 0; c < 3; ++c)
        result[pixel * 3 + c] = static_cast<float>(rgb[c]);
    }
  } else {
    cmsDoTransform(transform, device.data(), result.data(), pixels);
  }
  cmsDeleteTransform(transform);
  if (xyz)
    cmsCloseProfile(xyz);
  return result;
}

Stage Linear(int index) {
  std::array<double, 9> values;
  const auto matrix = scalar::Invert(connection::kSpaces[index].ToPcs());
  for (unsigned r = 0; r < 3; ++r)
    for (unsigned c = 0; c < 3; ++c)
      values[r * 3 + c] = matrix[r][c];
  return Matrix(values, {0, 0, 0}, false);
}
} // namespace

int main(int argc, char **argv) try {
  using namespace lut;
  namespace fs = std::filesystem;
  Check(argc == 3 && !fs::exists(argv[2]),
        "usage: lut_black IDENTITY_MPE_PROFILE NEW_OUTPUT_DIRECTORY");
  Check(cmsGetEncodedCMMversion() == 2190, "requires Little CMS 2.19");
  auto identity = cmsOpenProfileFromFile(argv[1], "r");
  Check(identity, "open identity MPE");
  const fs::path out(argv[2]);
  fs::create_directories(out);
  const auto profiles = BlackProfiles();
  std::ofstream manifest(out / "manifest.json");
  manifest << "{\"width\":17,\"height\":13,\"profiles\":[";
  constexpr unsigned pixels = 221;
  size_t components = 0, blacks = 0;
  for (size_t index = 0; index < profiles.size(); ++index) {
    const auto &profile = profiles[index];
    const auto &recipe = profile.recipe;
    Save(out / (recipe.name + ".icc"), profile.bytes);
    manifest << (index ? "," : "") << "{\"name\":\"" << recipe.name
             << "\",\"channels\":" << recipe.channels << '}';
    auto handle = cmsOpenProfileFromMem(
        profile.bytes.data(),
        static_cast<cmsUInt32Number>(profile.bytes.size()));
    Check(handle, "open native source LUT");
    std::vector<float> input;
    Bytes raw;
    for (unsigned pixel = 0; pixel < pixels; ++pixel)
      for (unsigned c = 0; c < recipe.channels; ++c) {
        const auto value =
            pixel < 2 ? static_cast<float>(pixel)
                      : static_cast<float>((pixel * 37 + c * 61) % 257) / 256;
        input.push_back(value);
        LE(raw, value);
      }
    Save(out / (recipe.name + ".f32le"), raw);
    for (unsigned intent = 0; intent < 4; ++intent) {
      const auto selected = intent == 3 ? 1 : intent;
      const bool compensate = intent == 0 || intent == 2;
      const auto primary_black = DetectBlack(profile, selected, false);
      const auto native_black = DetectBlack(profile, selected, true);
      if (compensate) {
        cmsCIEXYZ black{};
        const bool found = cmsDetectBlackPoint(&black, handle, intent, 0);
        Check(found == (recipe.channels != 5),
              "native darker-colorant availability");
        Bytes records;
        const double values[3] = {black.X, black.Y, black.Z};
        for (unsigned c = 0; c < 3; ++c) {
          Record(records, static_cast<float>(values[c]), primary_black[c],
                 native_black[c]);
          ++blacks;
        }
        Save(out / (recipe.name + "_black_" + std::to_string(intent) +
                    ".reference"),
             records);
      }
      for (int linear = -1;
           linear < static_cast<int>(connection::kSpaces.size()); ++linear) {
        const std::string target =
            linear < 0 ? "identity" : connection::kSpaces[linear].name;
        std::cerr << recipe.name << " to " << target << " intent " << intent
                  << '\n';
        const auto native =
            Native(handle, identity, recipe.channels, linear, input, intent);
        const auto connection = BlackConnection(
            primary_black, native_black,
            linear < 0 ? std::array<double, 3>{.00336, .0034731, .0028646}
                       : std::array<double, 3>{0, 0, 0});
        Bytes records;
        for (unsigned pixel = 0; pixel < pixels; ++pixel) {
          Values original;
          for (unsigned c = 0; c < recipe.channels; ++c)
            original.push_back({input[pixel * recipe.channels + c], 0});
          auto primary =
              Evaluate(profile.pipelines[selected * 2], original, false);
          auto native_bound =
              Evaluate(profile.pipelines[selected * 2], original, true);
          if (compensate) {
            primary = connection.apply(primary, false);
            native_bound = connection.apply(native_bound, true);
          }
          if (linear >= 0) {
            const auto matrix = Linear(linear);
            primary = matrix.apply(primary, false);
            native_bound = matrix.apply(native_bound, true);
          }
          for (unsigned c = 0; c < 3; ++c) {
            Record(records, native[pixel * 3 + c], primary[c], native_bound[c]);
            ++components;
          }
        }
        Save(out / (recipe.name + "_to_" + target + "_" +
                    std::to_string(intent) + ".reference"),
             records);
      }
    }
    cmsCloseProfile(handle);
  }
  manifest << "]}\n";
  Check(bool(manifest), "black corpus manifest");
  cmsCloseProfile(identity);
  std::cout << "Black-point profiles: " << profiles.size()
            << ", native black components: " << blacks
            << ", independent/native color components: " << components << '\n';
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
