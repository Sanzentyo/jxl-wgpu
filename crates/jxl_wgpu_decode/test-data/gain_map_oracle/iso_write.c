// Same private-entry-point adapter as iso_read.c, retaining libavif's complete writer source.
#include "src/write.c"
#include "iso_reference.h"

avifResult IsoWrite(const avifGainMap* metadata, avifRWData* output, avifDiagnostics* diag)
{
    avifRWStream stream;
    avifRWStreamStart(&stream, output);
    avifResult result = avifWriteGainmapMetadata(&stream, metadata, diag);
    avifRWStreamFinishWrite(&stream);
    return result;
}
