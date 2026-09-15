// Private weight selection from the complete pristine libavif gain-map implementation.
#include "src/gainmap.c"
#include "iso_reference.h"

float IsoWeight(float headroom, const avifGainMap* metadata)
{
    return avifGetGainMapWeight(headroom, metadata);
}
