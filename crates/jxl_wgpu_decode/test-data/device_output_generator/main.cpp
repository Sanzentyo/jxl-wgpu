// Native original JPEG XL samples and independent ICC device-output references.
#include "../../../jxl_wgpu/test-data/icc_generator/lut/jpeg_xl.hpp"
#include "../cmyk_generator/original.hpp"
#include <iterator>

namespace {
using namespace lut;
namespace fs = std::filesystem;
Bytes Read(const fs::path &path) {
  std::ifstream file(path, std::ios::binary);
  Check(bool(file), "read frozen corpus file");
  return Bytes(std::istreambuf_iterator<char>{file}, {});
}
Profile ProfileFor(const fs::path &base, Recipe recipe) {
  auto profile = Build(recipe);
  Check(profile.bytes == Read(base / (recipe.name + ".icc")),
        "independent recipe must reproduce the frozen profile");
  return profile;
}
Recipe Named(Format format, bool lab, unsigned channels) {
  const std::string kind = format == Format::Eight     ? "lut8"
                           : format == Format::Sixteen ? "lut16"
                                                       : "ab";
  return {kind + (lab ? "_lab_" : "_xyz_") + std::to_string(channels), format,
          lab, channels};
}
const std::array<Recipe, 6> kTargets{{
    {"lut16_xyz_1", Format::Sixteen, false, 1},
    {"lut16_channels2", Format::Sixteen, false, 2},
    {"ab_xyz_3", Format::AB, false, 3},
    {"ab_lab_4", Format::AB, true, 4},
    {"ab_channels5", Format::AB, true, 5},
    {"ab_channels15", Format::AB, false, 15},
}};

struct Generator {
  fs::path profiles, cmyk, out;
  std::ofstream manifest;
  unsigned cases = 0, components = 0;

  void Run(const Profile &source, unsigned mode, unsigned black) {
    const unsigned colors = source.recipe.channels;
    const bool ink = colors == 4;
    const unsigned frames = ink ? 3 : 1, stride = ink ? 6 : colors + 1;
    const std::string name =
        source.recipe.name + (ink         ? "_" + std::to_string(mode)
                              : mode == 0 ? "_modular"
                                          : "_vardct");
    const auto folder = ink ? cmyk : profiles / "decoder";
    const auto bytes = Read(folder / (name + ".jxl"));
    const auto original = ink ? cmyk::Decode(bytes, source.bytes)
                              : lut::Decode(bytes, source.bytes, colors);
    Check(original.size() == frames * image_pixels * stride,
          "original sample count");
    Bytes original_bytes;
    for (float value : original) {
      Check(std::isfinite(value), "finite native original samples");
      LE(original_bytes, value);
    }
    Check(original_bytes == Read(folder / (name + (ink ? ".f32" : ".f32le"))),
          "independent native decode must reproduce frozen original samples");
    auto target_recipe =
        kTargets[(cases + cases / kTargets.size()) % kTargets.size()];
    if (target_recipe.name == source.recipe.name)
      target_recipe = Named(Format::Sixteen, !source.recipe.lab, colors);
    const auto target = ProfileFor(profiles, target_recipe);
    auto src = cmsOpenProfileFromMem(source.bytes.data(), source.bytes.size());
    auto dst = cmsOpenProfileFromMem(target.bytes.data(), target.bytes.size());
    Check(src && dst, "native device output profiles");
    const auto input_format = cmsFormatterForColorspaceOfProfile(src, 4, TRUE);
    const auto output_format = cmsFormatterForColorspaceOfProfile(dst, 4, TRUE);
    Check(input_format && output_format, "native device component formatters");
    manifest << (cases ? "," : "") << "{\"name\":\"" << name
             << "\",\"source\":\"" << source.recipe.name << "\",\"target\":\""
             << target.recipe.name << "\",\"channels\":" << colors
             << ",\"target_channels\":" << target.recipe.channels
             << ",\"frames\":" << frames << ",\"mode\":" << mode
             << ",\"black\":" << black << '}';
    ++cases;
    for (unsigned spots = 0; spots < (ink ? 2u : 1u); ++spots)
      for (unsigned intent = 0; intent < 4; ++intent) {
        auto transform =
            cmsCreateTransform(src, input_format, dst, output_format, intent,
                               cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE);
        Check(transform, "native ICC device conversion");
        const unsigned selected = intent == 3 ? 1 : intent;
        Bytes records;
        for (unsigned p = 0; p < frames * image_pixels; ++p) {
          Values primary, native_bound;
          std::vector<float> input;
          for (unsigned c = 0; c < colors; ++c) {
            const double sample =
                original[p * stride + (ink && c == 3 ? 3 + black : c)];
            const double error = mode && (!ink || c < 3) ? 2e-5 : 0;
            const Value value =
                ink ? cmyk::Ink(sample, original[p * stride + 3 + (2 - black)],
                                c, spots, error)
                    : Value{sample, error};
            primary.push_back(value);
            native_bound.push_back({value.x, 0});
            input.push_back(static_cast<float>(value.x * (ink ? 100 : 1)));
          }
          std::vector<float> converted(target.recipe.channels);
          cmsDoTransform(transform, input.data(), converted.data(), 1);
          // Little CMS ink-space float formatters use percentages; the API uses
          // unit ink amounts.
          if (target.recipe.channels >= 4)
            for (auto &value : converted)
              value /= 100;
          for (const auto *pipeline : {&source.pipelines[selected * 2],
                                       &target.pipelines[selected * 2 + 1]})
            for (const auto &stage : pipeline->stages) {
              primary = stage.apply(primary, false);
              native_bound = stage.apply(native_bound, true);
            }
          for (unsigned c = 0; c < target.recipe.channels; ++c) {
            Record(records, converted[c], primary[c], native_bound[c]);
            ++components;
          }
        }
        cmsDeleteTransform(transform);
        Save(out / (name + "_" + std::to_string(spots) + "_" +
                    std::to_string(intent) + ".reference"),
             records);
      }
    cmsCloseProfile(src);
    cmsCloseProfile(dst);
    std::cerr << name << " -> " << target.recipe.name << '\n';
  }
};
} // namespace

int main(int argc, char **argv) try {
  Check(argc == 4 && !fs::exists(argv[3]),
        "usage: device_output_generator LUT_CORPUS CMYK_CORPUS "
        "NEW_OUTPUT_DIRECTORY");
  Check(JxlDecoderVersion() == 12000, "requires libjxl 0.12.0");
  Check(cmsGetEncodedCMMversion() == 2190, "requires Little CMS 2.19");
  fs::create_directories(argv[3]);
  Generator generator{argv[1], argv[2], argv[3],
                      std::ofstream(fs::path(argv[3]) / "manifest.json")};
  generator.manifest << "{\"width\":17,\"height\":9,\"cases\":[";
  for (auto format : {Format::Eight, Format::Sixteen, Format::AB})
    for (bool lab : {false, true}) {
      const auto source = ProfileFor(generator.profiles, Named(format, lab, 4));
      for (unsigned mode = 0; mode < 3; ++mode)
        generator.Run(source, mode, (mode + lab) % 2 == 0 ? 0 : 2);
    }
  for (auto format : {Format::Eight, Format::Sixteen, Format::AB})
    for (bool lab : {false, true})
      for (unsigned colors : {1u, 3u}) {
        const auto source =
            ProfileFor(generator.profiles, Named(format, lab, colors));
        for (unsigned mode = 0; mode < 2; ++mode)
          generator.Run(source, mode, 0);
      }
  generator.manifest << "]}\n";
  Check(bool(generator.manifest), "device output manifest");
  Check(generator.cases == 42, "device output source count");
  std::cout << generator.cases << " sources; " << generator.components
            << " independent/native components\n";
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
