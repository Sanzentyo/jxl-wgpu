// Offline JPEG XL original device samples, followed by independent ICC LUT
// equations.
#include "lut/jpeg_xl.hpp"
#include "lut/profile.hpp"
#include <iterator>

int main(int argc, char **argv) try {
  using namespace lut;
  namespace fs = std::filesystem;
  Check(argc == 3 && !fs::exists(argv[2]),
        "usage: lut_decoder LUT_CORPUS NEW_OUTPUT_DIRECTORY");
  Check(JxlEncoderVersion() == 12000 && JxlDecoderVersion() == 12000,
        "requires libjxl 0.12.0");
  Check(cmsGetEncodedCMMversion() == 2190, "requires Little CMS 2.19");
  const fs::path base(argv[1]), out(argv[2]);
  fs::create_directories(out);
  auto profile = [&](Format format, bool lab, unsigned channels) {
    const std::string kind = format == Format::Eight     ? "lut8"
                             : format == Format::Sixteen ? "lut16"
                                                         : "ab";
    const auto result =
        Build({kind + (lab ? "_lab_" : "_xyz_") + std::to_string(channels),
               format, lab, channels});
    std::ifstream file(base / (result.recipe.name + ".icc"), std::ios::binary);
    Check(bool(file), "read resident LUT corpus");
    const Bytes existing(std::istreambuf_iterator<char>{file}, {});
    Check(existing == result.bytes, "decoder/resident profile identity");
    return result;
  };
  std::ofstream manifest(out / "manifest.json");
  manifest << "{\"width\":17,\"height\":9,\"cases\":[";
  size_t cases = 0, components = 0;
  for (auto format : {Format::Eight, Format::Sixteen, Format::AB})
    for (bool lab : {false, true})
      for (unsigned channels : {1u, 3u}) {
        const auto source = profile(format, lab, channels);
        const auto target = profile(format == Format::Eight ? Format::Sixteen
                                    : format == Format::Sixteen ? Format::AB
                                                                : Format::Eight,
                                    !lab, channels == 1 ? 3 : 1);
        const auto &src = source.recipe;
        const auto &dst = target.recipe;
        auto source_handle = cmsOpenProfileFromMem(
            source.bytes.data(),
            static_cast<cmsUInt32Number>(source.bytes.size()));
        auto target_handle = cmsOpenProfileFromMem(
            target.bytes.data(),
            static_cast<cmsUInt32Number>(target.bytes.size()));
        Check(source_handle && target_handle, "open decoder LUT profiles");
        std::vector<float> input(image_pixels * (channels + 1));
        for (size_t i = 0; i < input.size(); ++i)
          input[i] = static_cast<float>(8 + (i * 37) % 101) / 128;
        for (bool modular : {true, false}) {
          const auto name = src.name + (modular ? "_modular" : "_vardct");
          std::cerr << name << " to " << dst.name << '\n';
          const auto bytes = Encode(source.bytes, channels, modular, input);
          Save(out / (name + ".jxl"), bytes);
          const auto native = Decode(bytes, source.bytes, channels);
          if (modular)
            Check(std::memcmp(native.data(), input.data(),
                              input.size() * sizeof(float)) == 0,
                  "lossless LUT source samples");
          Bytes original;
          std::vector<float> colors;
          for (unsigned pixel = 0; pixel < image_pixels; ++pixel)
            for (unsigned c = 0; c <= channels; ++c) {
              const auto value = native[pixel * (channels + 1) + c];
              Check(std::isfinite(value), "finite LUT device sample");
              LE(original, value);
              if (c < channels)
                colors.push_back(value);
              else
                Check(Bits(value) == Bits(input[pixel * (channels + 1) + c]),
                      "LUT alpha changed");
            }
          Save(out / (name + ".f32le"), original);
          manifest << (cases ? "," : "") << "{\"name\":\"" << name
                   << "\",\"source\":\"" << src.name << "\",\"target\":\""
                   << dst.name << "\",\"channels\":" << channels
                   << ",\"target_channels\":" << dst.channels
                   << ",\"modular\":" << (modular ? "true" : "false") << '}';
          ++cases;
          for (unsigned intent = 0; intent < 4; ++intent) {
            auto transform = cmsCreateTransform(
                source_handle, channels == 1 ? TYPE_GRAY_FLT : TYPE_RGB_FLT,
                target_handle, dst.channels == 1 ? TYPE_GRAY_FLT : TYPE_RGB_FLT,
                intent, cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE);
            Check(transform, "LUT decoder native transform");
            std::vector<float> converted(image_pixels * dst.channels);
            cmsDoTransform(transform, colors.data(), converted.data(),
                           image_pixels);
            cmsDeleteTransform(transform);
            const unsigned selected = intent == 3 ? 1 : intent;
            Bytes records;
            for (unsigned pixel = 0; pixel < image_pixels; ++pixel) {
              Values primary, native_bound;
              for (unsigned c = 0; c < channels; ++c) {
                const double x = colors[pixel * channels + c];
                primary.push_back({x, modular ? 0 : 2e-5});
                native_bound.push_back({x, 0});
              }
              // Both selected v4 LUT methods have the same PCS reference black
              // and exact media white, so the connection matrix is identity for
              // every intent.
              for (const auto *pipeline : {&source.pipelines[selected * 2],
                                           &target.pipelines[selected * 2 + 1]})
                for (const auto &stage : pipeline->stages) {
                  primary = stage.apply(primary, false);
                  native_bound = stage.apply(native_bound, true);
                }
              for (unsigned c = 0; c < dst.channels; ++c) {
                Record(records, converted[pixel * dst.channels + c], primary[c],
                       native_bound[c]);
                ++components;
              }
            }
            Save(out / (name + "_" + std::to_string(intent) + ".reference"),
                 records);
          }
        }
        cmsCloseProfile(source_handle);
        cmsCloseProfile(target_handle);
      }
  manifest << "]}\n";
  Check(bool(manifest), "LUT decoder manifest");
  std::cout << "JPEG XL LUT streams: " << cases
            << ", independent/native components: " << components << '\n';
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
