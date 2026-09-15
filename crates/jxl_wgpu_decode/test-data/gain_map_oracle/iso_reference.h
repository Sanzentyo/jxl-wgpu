// Offline adapters to the pinned libavif implementation, with no reimplementation of ISO syntax.
#ifndef JXL_WGPU_ISO_REFERENCE_H
#define JXL_WGPU_ISO_REFERENCE_H
#include <avif/avif.h>
#ifdef __cplusplus
extern "C" {
#endif
avifResult IsoRead(avifGainMap* metadata, const uint8_t* tmap, size_t size, avifDiagnostics* diag);
avifResult IsoWrite(const avifGainMap* metadata, avifRWData* output, avifDiagnostics* diag);
float IsoWeight(float headroom, const avifGainMap* metadata);
#ifdef __cplusplus
}
#endif
#endif
