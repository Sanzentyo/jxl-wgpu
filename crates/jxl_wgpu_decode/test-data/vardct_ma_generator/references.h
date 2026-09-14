#ifndef JXL_WGPU_VARDCT_MA_REFERENCES_H
#define JXL_WGPU_VARDCT_MA_REFERENCES_H

#include <jxl/memory_manager.h>
#include <filesystem>
#include "lib/jxl/base/status.h"

jxl::Status WriteReferences(JxlMemoryManager* memory,
                            const std::filesystem::path& directory);

#endif
