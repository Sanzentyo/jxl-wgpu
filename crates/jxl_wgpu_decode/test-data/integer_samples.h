/* Public libjxl integer encoding supports up to 24 bits. These independent declarations
 * exercise wide sample normalization through both producers and the shared render graph. */
static const uint32_t integer_extra_bits[] = {17, 18, 19, 20, 21, 22, 23, 24, 18};

static float integer_sample(uint32_t x, uint32_t y, uint32_t channel, uint32_t frame, uint32_t bits) {
  uint32_t maximum = (1u << bits) - 1;
  uint32_t value = x * 0x9e3779b9u + y * 0x85ebca6bu + channel * 0xc2b2ae35u + frame * 0x27d4eb2du;
  value ^= value >> 16; value &= maximum;
  if (x % 11 == 0) value = 0;
  if (x % 11 == 1) value = maximum;
  return (float)((double)value / (double)maximum);
}
