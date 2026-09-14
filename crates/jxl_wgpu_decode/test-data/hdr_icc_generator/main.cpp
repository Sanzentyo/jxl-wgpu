// Independent HDR PCS references connect to exact ICC profiles through native and scalar CMMs.
#include <icc/rgb.hpp>
using namespace rgb_icc;

int main(int argc, char **argv) try {
  namespace fs = std::filesystem;
  Check(argc == 4 && !fs::exists(argv[3]),
        "usage: hdr-icc ICC_ROOT PCS_INPUT NEW_OUTPUT");
  Check(cmsGetEncodedCMMversion() == 2190, "requires Little CMS 2.19");
  const fs::path root(argv[1]), input(argv[2]), output(argv[3]);
  fs::create_directories(output);
  std::ifstream list(input / "manifest.tsv");
  Check(bool(list), "source manifest");
  std::ofstream manifest(output / "manifest.json");
  manifest << "{\"cases\":[";
  std::string name, profile;
  unsigned width, height, frames;
  size_t cases = 0, components = 0;
  while (list >> name >> profile >> width >> height >> frames) {
    const auto spec =
        std::find_if(kTargets.begin(), kTargets.end(),
                     [&](const auto &spec) { return spec.name == profile; });
    Check(spec != kTargets.end(), "declared target method");
    Target target(root, *spec);
    const auto pixels = Sources(input / (name + ".pcs"));
    Check(pixels.size() == size_t{width} * height * frames, "presentation dimensions");
    manifest << (cases ? "," : "") << "{\"name\":\"" << name
             << "\",\"target\":\"" << profile
             << "\",\"channels\":" << target.channels
             << ",\"width\":" << width << ",\"height\":" << height
             << ",\"frames\":" << frames << '}';
    for (unsigned intent = 0; intent < 4; ++intent) {
      const auto native = target.Native(pixels, intent);
      Bytes records;
      for (size_t p = 0; p < pixels.size(); ++p) {
        const auto exact = target.Evaluate(pixels[p], intent, false);
        auto center = pixels[p];
        for (auto &value : center)
          value.radius = 0;
        const auto bound = target.Evaluate(center, intent, true);
        for (unsigned c = 0; c < target.channels; ++c) {
          try {
            Record(records, native[p * target.channels + c], exact[c],
                   bound[c]);
          } catch (...) {
            std::cerr << name << " -> " << profile << " intent " << intent
                      << " pixel " << p << " channel " << c << '\n';
            throw;
          }
          ++components;
        }
      }
      Save(output / (name + "_" + std::to_string(intent) + ".reference"),
           records);
    }
    std::cout << name << " -> " << profile << '\n';
    ++cases;
  }
  Check(list.eof() && cases == 56, "complete 56-case HDR source manifest");
  manifest << "]}\n";
  Check(bool(manifest), "output manifest");
  std::cout << cases << " streams, " << components
            << " independent/native color components\n";
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
