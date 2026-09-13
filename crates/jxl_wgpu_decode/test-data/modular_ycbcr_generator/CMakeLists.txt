cmake_minimum_required(VERSION 3.20)
project(jxl_wgpu_modular_ycbcr_fixtures LANGUAGES C CXX)

option(JXL_SCALAR_ONLY "Build the independent oracle without SIMD inverse transforms" OFF)
if(JXL_SCALAR_ONLY)
  add_compile_definitions(HWY_COMPILE_ONLY_SCALAR=1)
endif()

set(JXL_SOURCE "" CACHE PATH "libjxl source checkout used for fixture generation")
execute_process(COMMAND git -C "${JXL_SOURCE}" rev-parse HEAD
  OUTPUT_VARIABLE JXL_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT JXL_COMMIT STREQUAL "a7a9c787341cf703dede03c2009fa460cae5e5df")
  message(FATAL_ERROR "This fixture generator requires the pinned libjxl 0.12.0 commit")
endif()

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
foreach(feature TOOLS DEVTOOLS DOXYGEN MANPAGES BENCHMARK EXAMPLES JNI SJPEG OPENEXR SKCMS VIEWERS PLUGINS TCMALLOC)
  set(JPEGXL_ENABLE_${feature} OFF CACHE BOOL "" FORCE)
endforeach()
set(JPEGXL_FORCE_SYSTEM_HWY ON CACHE BOOL "" FORCE)
set(JPEGXL_FORCE_SYSTEM_BROTLI ON CACHE BOOL "" FORCE)
set(JPEGXL_FORCE_SYSTEM_LCMS2 ON CACHE BOOL "" FORCE)
add_subdirectory("${JXL_SOURCE}" libjxl EXCLUDE_FROM_ALL)

add_executable(generate_modular_ycbcr main.cpp cases.cpp)
target_compile_features(generate_modular_ycbcr PRIVATE cxx_std_17)
target_compile_options(generate_modular_ycbcr PRIVATE -Wall -Wextra -Werror)
# Keep strict diagnostics for this generator while treating upstream headers as dependencies.
target_include_directories(generate_modular_ycbcr SYSTEM PRIVATE "${JXL_SOURCE}")
target_link_libraries(generate_modular_ycbcr PRIVATE jxl-internal)

if(JXL_SCALAR_ONLY)
  add_executable(decode_modular_ycbcr_scalar ../decode_extra_channels.c)
  target_link_libraries(decode_modular_ycbcr_scalar PRIVATE jxl-internal jxl_cms)
endif()
