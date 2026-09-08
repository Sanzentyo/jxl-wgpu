/* Finite dyadic samples and independently declared mixed extra-channel precision. */
static const uint32_t floating_extra_bits[] = {16, 7, 32, 24, 16, 24, 6, 32, 16};
static const uint32_t floating_extra_exponents[] = {5, 0, 8, 7, 8, 8, 0, 8, 5};

static uint32_t floating_exponent(uint32_t bits) {
  return bits == 16 ? 5 : bits == 24 ? 7 : 8;
}

static float floating_sample(uint32_t x, uint32_t y, uint32_t channel, uint32_t frame, int color) {
  if (x % 11 == 0) return 0;
  if (x % 11 == 1) return 1;
  int value = (int)((193*x + 317*y + 97*channel + (x^y)*(23+channel) + frame*(71+channel)) % (color ? 193 : 129));
  return (float)(value - (color ? 32 : 0)) / 128.0f;
}
