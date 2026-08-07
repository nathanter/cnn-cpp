// does putting fil_size to a compile-time constant let
// the compiler unroll the 3x3 filter loops, and what is that worth?
//
// Both versions below are copies of Conv_Layer2D::feed_forward with identical
// arithmetic. The ONLY difference is how the filter-loop bound is expressed.
// Everything else -- variable names, loop structure, Tensor calls -- is verbatim
// from src/layers.hpp so the comparison stays faithful.
//
// Build:
//   g++ -std=c++17 -O3 -Isrc tests/compiler-opt-test.cpp -o compiler-opt-test
//   ./compiler-opt-test
#include "tensor.hpp"
#include <chrono>
#include <cstdio>
#include <stdexcept>

static unsigned int lcg_state = 12345u;
static float next_val() {
  lcg_state = lcg_state * 1103515245u + 12345u;
  return static_cast<float>((lcg_state >> 16) & 0x7fff) / 32768.0f - 0.5f;
}

// ---------------------------------------------------------------------------
// VERSION A -- runtime bound (what layers.hpp does today)
//
// The filter loops read this->fil_size, a mutable public member. The compiler
// must load it from memory and cannot know its value, so:
//   - the trip count is unknown  -> the 3x3 stays a real loop
//   - f1 * fil_size stays a real multiply, recomputed every tap
// ---------------------------------------------------------------------------
struct ConvRuntimeBound {
  Tensor data;
  Tensor biases_data;
  int nodes;
  int fil_size = 3;
  Tensor last_input;

  ConvRuntimeBound(int neurons) : data(neurons, 3, 3), biases_data(neurons) {
    this->nodes = neurons;
  }

  Tensor feed_forward(Tensor &input) {
    this->last_input = input;
    Tensor output(this->nodes, input.dimen_list[0] - 2,
                  input.dimen_list[1] - 2);

    int out_index = 0;
    for (int node = 0; node < this->nodes; node++) {
      int start_of_filter_in_conv_weights =
          this->data.flat_from_indexes(node, 0, 0);

      for (int ax1 = 0; ax1 <= input.dimen_list[0] - 3; ax1++) {
        for (int ax2 = 0; ax2 <= input.dimen_list[1] - 3; ax2++) {
          float sum = 0.00;
          int start_of_filter_in_input =
              ax1 * input.indexing_list[0] + ax2 * input.indexing_list[1];

          // bound is the member -> unknown at compile time
          for (int f1 = 0; f1 < fil_size; f1++) {
            int place_in_input_tensor =
                input.indexing_list[0] * f1 + start_of_filter_in_input;
            for (int f2 = 0; f2 < fil_size; f2++) {
              sum += static_cast<float>(
                  input.get_element_flat(place_in_input_tensor + f2) *
                  this->data.get_element_flat(
                      start_of_filter_in_conv_weights + f1 * fil_size + f2));
            }
          }

          sum += biases_data.get_element_flat(node);
          output.flat_assign(out_index, sum);
          out_index++;
        }
      }
    }
    return output;
  }
};

// ---------------------------------------------------------------------------
// VERSION B -- pinned bound
//
// Two changes, both required:
//   1. Copy the member into a LOCAL. The local cannot be modified but Pinning
//      this->fil_size directly does NOT work -- the value gets reloaded.
//   2. Throw if it isn't 3. the compiler can prove F == 3 so it stops writing loops and just writes out the instructions for each loop. 
//.     A local copy WITHOUT
//      this check buys nothing, because stable != known.
//
//
// Result: trip count is 3, so the 3x3 flattens into 9 straight-line multiply-
// accumulates and f1 * F + f2 folds to the literals 0..8.
// ---------------------------------------------------------------------------
struct ConvPinnedBound {
  Tensor data;
  Tensor biases_data;
  int nodes;
  int fil_size = 3;
  Tensor last_input;

  ConvPinnedBound(int neurons) : data(neurons, 3, 3), biases_data(neurons) {
    this->nodes = neurons;
  }

  Tensor feed_forward(Tensor &input) {
    this->last_input = input;

    const int F = this->fil_size;                          // (1) local copy
    if (F != 3) throw std::runtime_error("filter must be 3x3");  // (2) pin it

    Tensor output(this->nodes, input.dimen_list[0] - 2,
                  input.dimen_list[1] - 2);

    int out_index = 0;
    for (int node = 0; node < this->nodes; node++) {
      int start_of_filter_in_conv_weights =
          this->data.flat_from_indexes(node, 0, 0);

      for (int ax1 = 0; ax1 <= input.dimen_list[0] - 3; ax1++) {
        for (int ax2 = 0; ax2 <= input.dimen_list[1] - 3; ax2++) {
          float sum = 0.00;
          int start_of_filter_in_input =
              ax1 * input.indexing_list[0] + ax2 * input.indexing_list[1];

          // every fil_size below is now F -- missing even one of them leaves a
          // member load in the inner loop and gives back part of the win
          for (int f1 = 0; f1 < F; f1++) {
            int place_in_input_tensor =
                input.indexing_list[0] * f1 + start_of_filter_in_input;
            for (int f2 = 0; f2 < F; f2++) {
              sum += static_cast<float>(
                  input.get_element_flat(place_in_input_tensor + f2) *
                  this->data.get_element_flat(
                      start_of_filter_in_conv_weights + f1 * F + f2));
            }
          }

          sum += biases_data.get_element_flat(node);
          output.flat_assign(out_index, sum);
          out_index++;
        }
      }
    }
    return output;
  }
};

// Fills a layer with the same deterministic weights regardless of which version
// it is, so both compute identical output.
template <class Conv> Conv make_layer(int neurons) {
  lcg_state = 12345u;
  Conv layer(neurons);
  for (int i = 0; i < layer.data.size; i++) layer.data.data[i] = next_val();
  for (int i = 0; i < layer.biases_data.size; i++)
    layer.biases_data.data[i] = next_val();
  return layer;
}

template <class Conv>
double time_version(const char *name, Tensor &image, float &sink,
                    double *checksum) {
  const int kIters = 2000;
  Conv layer = make_layer<Conv>(20);

  Tensor warm = layer.feed_forward(image);
  double sum = 0.0;
  for (int i = 0; i < warm.size; i++) sum += warm.data[i];
  *checksum = sum;

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIters; i++) {
    Tensor out = layer.feed_forward(image);
    sink += out.data[0];
  }
  auto t1 = std::chrono::steady_clock::now();

  double us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
  printf("  %-34s %8.2f us/image\n", name, us);
  return us;
}

int main() {
  // fake input: deterministic 28x28 "image", same values every run
  lcg_state = 999u;
  Tensor image(28, 28);
  for (int i = 0; i < image.size; i++) image.data[i] = next_val();

  float sink = 0.0f;
  double checksum_a = 0.0, checksum_b = 0.0;

  // First measurement of the process runs cold (cache + CPU frequency ramp),
  // so do the whole comparison twice and read the second block.
  for (int pass = 0; pass < 2; pass++) {
    printf("pass %d%s\n", pass + 1, pass == 0 ? " (cold -- ignore)" : "");
    double a = time_version<ConvRuntimeBound>("A: runtime bound (fil_size)",
                                              image, sink, &checksum_a);
    double b = time_version<ConvPinnedBound>("B: pinned bound (const F + throw)",
                                             image, sink, &checksum_b);
    printf("  -> speedup %.2fx\n\n", a / b);
  }

  // Correctness: the optimization must not change what the network computes.
  double diff = checksum_a - checksum_b;
  if (diff < 0) diff = -diff;
  printf("output checksum  A=%.6f  B=%.6f  diff=%g -> %s\n", checksum_a,
         checksum_b, diff, diff < 1e-3 ? "MATCH" : "MISMATCH");

  printf("(sink %g)\n", sink);
  return diff < 1e-3 ? 0 : 1;
}
