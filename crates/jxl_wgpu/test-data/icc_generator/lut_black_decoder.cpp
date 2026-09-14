// Original JPEG XL samples followed by independent v2-to-v4 LUT connections.
#include "lut/black.hpp"
#include "lut/jpeg_xl.hpp"
#include <iterator>

int main(int argc, char **argv) try {
  using namespace lut;
  namespace fs = std::filesystem;
  Check(argc == 3 && !fs::exists(argv[2]),
        "usage: lut_black_decoder BLACK_CORPUS NEW_OUTPUT_DIRECTORY");
  Check(JxlEncoderVersion() == 12000 && JxlDecoderVersion() == 12000,
        "requires libjxl 0.12.0");
  Check(cmsGetEncodedCMMversion() == 2190, "requires Little CMS 2.19");
  const fs::path base(argv[1]), out(argv[2]);
  fs::create_directories(out);
  std::ofstream manifest(out / "manifest.json");
  manifest << "{\"width\":17,\"height\":9,\"cases\":[";
  size_t cases = 0, components = 0;
  for (const auto &source : BlackProfiles()) {
    const auto &src = source.recipe;
    const auto channels = src.channels;
    if (channels != 1 && channels != 3)
      continue;
    std::ifstream file(base / (src.name + ".icc"), std::ios::binary);
    Check(bool(file), "read black corpus profile");
    const Bytes existing(std::istreambuf_iterator<char>{file}, {});
    Check(existing == source.bytes, "decoder/resident source profile identity");
    const auto target = Build(
        {"target_" + src.name, Format::AB, !src.lab, channels == 1 ? 3u : 1u});
    const auto &dst = target.recipe;
    Save(out / (dst.name + ".icc"), target.bytes);
    auto source_handle = cmsOpenProfileFromMem(
        source.bytes.data(), static_cast<cmsUInt32Number>(source.bytes.size()));
    auto target_handle = cmsOpenProfileFromMem(
        target.bytes.data(), static_cast<cmsUInt32Number>(target.bytes.size()));
    Check(source_handle && target_handle, "open decoder black-point profiles");
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
              "lossless black-point source samples");
      Bytes original;
      std::vector<float> colors;
      for (unsigned pixel = 0; pixel < image_pixels; ++pixel)
        for (unsigned c = 0; c <= channels; ++c) {
          const auto value = native[pixel * (channels + 1) + c];
          Check(std::isfinite(value), "finite black-point source sample");
          LE(original, value);
          if (c < channels)
            colors.push_back(value);
          else
            Check(Bits(value) == Bits(input[pixel * (channels + 1) + c]),
                  "black-point alpha changed");
        }
      Save(out / (name + ".f32le"), original);
      manifest << (cases ? "," : "") << "{\"name\":\"" << name
               << "\",\"source\":\"" << src.name << "\",\"target\":\"decoder/"
               << dst.name << "\",\"channels\":" << channels
               << ",\"target_channels\":" << dst.channels
               << ",\"modular\":" << (modular ? "true" : "false") << '}';
      ++cases;
      for (unsigned intent = 0; intent < 4; ++intent) {
        auto transform = cmsCreateTransform(
            source_handle, channels == 1 ? TYPE_GRAY_FLT : TYPE_RGB_FLT,
            target_handle, dst.channels == 1 ? TYPE_GRAY_FLT : TYPE_RGB_FLT,
            intent, cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE);
        Check(transform, "native decoder black-point conversion");
        std::vector<float> converted(image_pixels * dst.channels);
        cmsDoTransform(transform, colors.data(), converted.data(),
                       image_pixels);
        cmsDeleteTransform(transform);
        const auto selected = intent == 3 ? 1 : intent;
        const auto connection = BlackConnection(
            DetectBlack(source, selected, false),
            DetectBlack(source, selected, true), {.00336, .0034731, .0028646});
        Bytes records;
        for (unsigned pixel = 0; pixel < image_pixels; ++pixel) {
          Values primary, native_bound;
          for (unsigned c = 0; c < channels; ++c) {
            const double x = colors[pixel * channels + c];
            primary.push_back({x, modular ? 0 : 2e-5});
            native_bound.push_back({x, 0});
          }
          primary = Evaluate(source.pipelines[selected * 2], primary, false);
          native_bound =
              Evaluate(source.pipelines[selected * 2], native_bound, true);
          if (intent == 0 || intent == 2) {
            primary = connection.apply(primary, false);
            native_bound = connection.apply(native_bound, true);
          }
          primary =
              Evaluate(target.pipelines[selected * 2 + 1], primary, false);
          native_bound =
              Evaluate(target.pipelines[selected * 2 + 1], native_bound, true);
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
  Check(bool(manifest), "decoder black-point manifest");
  std::cout << "JPEG XL black-point streams: " << cases
            << ", independent/native components: " << components << '\n';
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
