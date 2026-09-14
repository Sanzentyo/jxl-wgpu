#pragma once
#include <jxl/color_encoding.h>
#include <jxl/decode.h>
#include <cstdint>
#include <filesystem>
#include <vector>

void Require(bool condition, const char* operation);
std::vector<float> Decode(const std::vector<uint8_t>& encoded,
                         JxlColorEncoding original, uint32_t nits, bool linear);
void WriteBytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes);
void WriteFloats(const std::filesystem::path& path, const std::vector<float>& values);
