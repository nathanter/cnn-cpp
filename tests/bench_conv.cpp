// Times conv feed_forward + backprop in isolation, deterministic input.
//
// Build:
//   g++ -std=c++17 -O3 -Isrc tests/bench_conv.cpp -o bench_conv
//   ./bench_conv
//
// The first measurement in a process runs cold -- process startup, cold
// caches, CPU frequency ramp -- and reads high enough to mislead. forward
// came out anywhere from 12 to 24 us/image on IDENTICAL code depending on
// what the machine was doing, and since forward is timed first it absorbs all
// of it. So the whole thing runs twice and only pass 2 means anything, same
// structure as compiler-opt-test.cpp.
#include "layers.hpp"
#include <chrono>
#include <cstdio>

static unsigned int lcg_state = 12345u;
static float next_val() {
  lcg_state = lcg_state * 1103515245u + 12345u;
  return static_cast<float>((lcg_state >> 16) & 0x7fff) / 32768.0f - 0.5f;
}

int main() {
  const int kIters = 2000;

  lcg_state = 999u;
  Tensor image(28, 28);
  for (int i = 0; i < image.size; i++) image.data[i] = next_val();


  Tensor grad(20, 26, 26);
  for (int i = 0; i < grad.size; i++) grad.data[i] = next_val() * 0.001f;

  float sink = 0.0f;

  for (int pass = 0; pass < 2; pass++) {
    printf("pass %d%s\n", pass + 1, pass == 0 ? " (cold -- ignore)" : "");

    // Fresh layer each pass. backprop is gradient descent, so it mutates the
    // filters -- without this, pass 2 would start on weights 2000 updates
    // drifted from where pass 1 started and the two would not be comparable.
    Conv_Layer2D conv(20);
    lcg_state = 12345u;
    for (int i = 0; i < conv.data.size; i++) conv.data.data[i] = next_val();
    for (int i = 0; i < conv.biases_data.size; i++)
      conv.biases_data.data[i] = next_val();

    // warm both paths before either clock starts. feed_forward has to run
    // first regardless -- it is what sets last_input for backprop.
    Tensor warm = conv.feed_forward(image);
    sink += warm.data[0];
    conv.backprop(grad);

    // forward
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; i++) {
      Tensor out = conv.feed_forward(image);
      sink += out.data[0];
    }
    auto t1 = std::chrono::steady_clock::now();

    // backward
    for (int i = 0; i < kIters; i++) conv.backprop(grad);
    auto t2 = std::chrono::steady_clock::now();

    double fwd = std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
    double bwd = std::chrono::duration<double, std::micro>(t2 - t1).count() / kIters;
    printf("  forward  %8.2f us/image\n  backward %8.2f us/image\n"
           "  total    %8.2f us/image\n\n",
           fwd, bwd, fwd + bwd);
  }

  printf("(sink %g)\n", sink);
  return 0;
}
