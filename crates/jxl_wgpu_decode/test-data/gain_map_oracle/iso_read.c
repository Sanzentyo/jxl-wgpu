// The ISO parser is private. Compile the entire pristine translation unit and expose only its
// tmap entry point; the static library's read.c object is consequently not pulled into the link.
#include "src/read.c"
#include "iso_reference.h"

avifResult IsoRead(avifGainMap* metadata, const uint8_t* tmap, size_t size, avifDiagnostics* diag)
{
    return avifParseToneMappedImageBox(metadata, tmap, size, diag);
}
