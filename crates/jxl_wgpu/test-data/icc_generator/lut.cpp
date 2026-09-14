// Offline ICC.1:2022 LUT equations and Little CMS 2.19 references.
// Neither production Rust/WGSL nor GPU-produced pixels are used by this
// generator.
#include "lut/profile.hpp"

static void Diagnose(cmsHPROFILE profile, const lut::Recipe &recipe,
                     const lut::Pipeline &pipeline, unsigned intent,
                     bool reverse, const lut::Values &input) {
  using namespace lut;
  auto value = input;
  std::cerr << "independent stages:\n";
  for (const auto &stage : pipeline.stages) {
    value = stage.apply(value, false);
    for (auto v : value)
      std::cerr << v.x << ' ';
    std::cerr << '\n';
  }
  auto encoded = input;
  if (reverse)
    for (const auto &stage :
         Pcs(recipe.lab, recipe.format == Format::Sixteen, true))
      encoded = stage.apply(encoded, false);
  std::vector<float> native;
  for (auto v : encoded)
    native.push_back(static_cast<float>(v.x));
  const auto tag = static_cast<cmsTagSignature>(
      (reverse ? cmsSigBToA0Tag : cmsSigAToB0Tag) + (intent == 3 ? 1 : intent));
  auto raw = static_cast<cmsPipeline *>(cmsReadTag(profile, tag));
  Check(raw, "diagnostic native pipeline");
  std::cerr << "native stages (without PCS wrapper):\n";
  for (auto stage = cmsPipelineGetPtrToFirstStage(raw); stage;
       stage = cmsStageNext(stage)) {
    auto one = cmsPipelineAlloc(nullptr, cmsStageInputChannels(stage),
                                cmsStageOutputChannels(stage));
    Check(one, "diagnostic stage pipeline");
    Check(cmsPipelineInsertStage(one, cmsAT_END, cmsStageDup(stage)),
          "diagnostic stage clone");
    std::vector<float> output(cmsStageOutputChannels(stage));
    cmsPipelineEvalFloat(native.data(), output.data(), one);
    cmsPipelineFree(one);
    native = output;
    for (auto v : output)
      std::cerr << v << ' ';
    std::cerr << '\n';
  }
}

int main(int argc, char **argv) try {
  using namespace lut;
  namespace fs = std::filesystem;
  Check(argc == 3 && !fs::exists(argv[2]),
        "usage: lut IDENTITY_MPE_PROFILE NEW_OUTPUT_DIRECTORY");
  Check(cmsGetEncodedCMMversion() == 2190, "requires Little CMS 2.19");
  const fs::path out(argv[2]);
  fs::create_directories(out);
  auto identity = cmsOpenProfileFromFile(argv[1], "r");
  Check(identity, "open identity MPE");
  std::vector<Recipe> recipes;
  for (auto format : {Format::Eight, Format::Sixteen, Format::AB})
    for (bool lab : {false, true})
      for (unsigned channels : {1u, 3u, 4u}) {
        const std::string kind = format == Format::Eight     ? "lut8"
                                 : format == Format::Sixteen ? "lut16"
                                                             : "ab";
        recipes.push_back(
            {kind + (lab ? "_lab_" : "_xyz_") + std::to_string(channels),
             format, lab, channels});
      }
  for (bool lab : {false, true}) {
    recipes.push_back({std::string("ab_b_") + (lab ? "lab" : "xyz"), Format::AB,
                       lab, 3, 4, false, false});
    recipes.push_back({std::string("ab_matrix_") + (lab ? "lab" : "xyz"),
                       Format::AB, lab, 3, 4, true, false});
    for (unsigned channels : {1u, 3u, 4u})
      recipes.push_back({std::string("ab_clut_") + (lab ? "lab_" : "xyz_") +
                             std::to_string(channels),
                         Format::AB, lab, channels, 4, false, true});
  }
  recipes.push_back(
      {"ab_shared_rgb", Format::AB, false, 3, 4, true, true, true});
  recipes.push_back(
      {"ab_shared_gray", Format::AB, true, 1, 4, true, true, true});
  recipes.push_back({"ab_channels15", Format::AB, false, 15});
  recipes.push_back({"ab_channels5", Format::AB, true, 5});
  recipes.push_back({"lut16_channels2", Format::Sixteen, false, 2});
  for (auto format : {Format::Eight, Format::Sixteen})
    for (bool lab : {false, true})
      for (unsigned channels : {1u, 3u}) {
        recipes.push_back(
            {std::string(format == Format::Eight ? "lut8_v2_" : "lut16_v2_") +
                 (lab ? "lab_" : "xyz_") + std::to_string(channels),
             format, lab, channels, 2});
      }
  constexpr unsigned pixels = 221;
  const std::vector<float> edges{0,
                                 std::nextafter(0.f, 1.f),
                                 1.f / 65535,
                                 .03125f,
                                 .125f,
                                 .25f,
                                 std::nextafter(.25f, 0.f),
                                 std::nextafter(.25f, 1.f),
                                 .5f,
                                 .75f,
                                 std::nextafter(1.f, 0.f),
                                 1};
  std::ofstream manifest(out / "manifest.json");
  manifest << "{\"width\":17,\"height\":13,\"profiles\":[";
  size_t components = 0;
  for (size_t index = 0; index < recipes.size(); ++index) {
    const auto profile = Build(recipes[index]);
    const auto &recipe = profile.recipe;
    Save(out / (recipe.name + ".icc"), profile.bytes);
    manifest << (index ? "," : "") << "{\"name\":\"" << recipe.name
             << "\",\"channels\":" << recipe.channels << '}';
    auto handle = cmsOpenProfileFromMem(
        profile.bytes.data(),
        static_cast<cmsUInt32Number>(profile.bytes.size()));
    Check(handle, "open native LUT profile");
    const auto format = cmsFormatterForColorspaceOfProfile(handle, 4, TRUE);
    Check(format != 0, "native LUT device format");
    for (bool reverse : {false, true}) {
      const unsigned input_channels = reverse ? 3 : recipe.channels,
                     output_channels = reverse ? recipe.channels : 3;
      std::vector<float> input;
      for (unsigned pixel = 0; pixel < pixels; ++pixel) {
        std::vector<double> values(input_channels);
        for (unsigned c = 0; c < input_channels; ++c)
          values[c] =
              pixel < edges.size()
                  ? edges[(pixel + c * 5) % edges.size()]
                  : static_cast<float>(((pixel * (17 + c * 8) + c * 13) % 101) /
                                       100.0);
        if (reverse && recipe.lab) {
          const auto xyz = LabValue(
              {100 * values[0], 255 * values[1] - 128, 255 * values[2] - 128},
              true);
          values.assign(xyz.begin(), xyz.end());
        } else if (reverse)
          for (auto &value : values)
            value *= 1.5;
        for (double value : values)
          input.push_back(static_cast<float>(value));
      }
      Bytes raw;
      for (float value : input)
        LE(raw, value);
      Save(out / (recipe.name + (reverse ? "_reverse" : "_forward") + ".f32le"),
           raw);
      auto native_input = input;
      // CMYK and 5CLR..FCLR float formatters use percentages; RGB/Gray/2CLR
      // use unit values (Little CMS 2.19 cmspack.c, IsInkSpace).
      if (!reverse && recipe.channels >= 4)
        for (auto &value : native_input)
          value *= 100;
      for (unsigned intent = 0; intent < 4; ++intent) {
        // The production connection rejects automatic v2-LUT to v4 black-point
        // compensation until its metadata program can execute on the GPU.
        if (!reverse && recipe.version == 2 && (intent == 0 || intent == 2))
          continue;
        std::cerr << recipe.name << " intent " << intent << " reverse "
                  << reverse << '\n';
        auto transform = cmsCreateTransform(
            reverse ? identity : handle, reverse ? TYPE_RGB_FLT : format,
            reverse ? handle : identity, reverse ? format : TYPE_RGB_FLT,
            intent, cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE);
        Check(transform, "native LUT transform");
        std::vector<float> native(pixels * output_channels);
        cmsDoTransform(transform, native_input.data(), native.data(), pixels);
        cmsDeleteTransform(transform);
        if (reverse && recipe.channels >= 4)
          for (auto &value : native)
            value /= 100;
        const auto &pipeline =
            profile.pipelines[(intent == 3 ? 1 : intent) * 2 + reverse];
        Bytes records;
        for (unsigned pixel = 0; pixel < pixels; ++pixel) {
          Values primary;
          for (unsigned c = 0; c < input_channels; ++c)
            primary.push_back({input[pixel * input_channels + c], 0});
          auto native_bound = primary;
          for (const auto &stage : pipeline.stages) {
            primary = stage.apply(primary, false);
            native_bound = stage.apply(native_bound, true);
          }
          for (unsigned c = 0; c < output_channels; ++c) {
            try {
              Record(records, native[pixel * output_channels + c], primary[c],
                     native_bound[c]);
            } catch (const std::exception &) {
              Values original;
              for (unsigned axis = 0; axis < input_channels; ++axis)
                original.push_back({input[pixel * input_channels + axis], 0});
              std::cerr << "pixel " << pixel << " component " << c << '\n';
              Diagnose(handle, recipe, pipeline, intent, reverse, original);
              throw;
            }
            ++components;
          }
        }
        Save(out / (recipe.name + (reverse ? "_reverse_" : "_forward_") +
                    std::to_string(intent) + ".reference"),
             records);
      }
    }
    cmsCloseProfile(handle);
  }
  manifest << "]}\n";
  Check(bool(manifest), "LUT manifest");
  cmsCloseProfile(identity);
  std::cout << "LUT profiles: " << recipes.size()
            << ", independent/native components: " << components << '\n';
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
