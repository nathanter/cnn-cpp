// What did the backprop optimization actually buy?
//
// Two versions, both verbatim copies of Conv_Layer2D::backprop -- A is what it
// looked like before commit af20383, B is what it looks like now. Same
// arithmetic, same accumulation order, so the gradients match to the bit and
// the only difference in the timing is the code shape.
//
// Build:
//   g++ -std=c++17 -O3 -Isrc tests/backprop-opt-test.cpp -o backprop-opt-test
//   ./backprop-opt-test
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
// VERSION A -- the old version (pre af20383)
//
//   - filter loops read this->fil_size, so the 3x3 stays a real loop and
//     f1 * fil_size stays a real multiply
//   - dl_dfilter is copy-constructed from data and then zeroed, so 9*nodes
//     floats get copied out just to be immediately overwritten
//   - flat_from_indexes(x, 0, 0) is recomputed inside the ax2 loop, and
//     flat_from_indexes(ax1 + f1, ax2) is recomputed inside the f1 loop
// ---------------------------------------------------------------------------
struct ConvBackpropOld {
  Tensor data;
  Tensor biases_data;
  int nodes;
  int fil_size = 3;
  Tensor last_input;
  float learning_rate = .01f;

  ConvBackpropOld(int neurons) : data(neurons, 3, 3), biases_data(neurons) {
    this->nodes = neurons;
  }

  void backprop(Tensor &losses) {
    Tensor dl_dfilter = data;
    dl_dfilter.zeros();

    for (int x = 0; x < nodes; x++) {
      float loss_sum = 0.0f;
      for (int ax1 = 0; ax1 < losses.dimen_list[1]; ax1++) {
        int current_start_of_row = losses.flat_from_indexes(x, ax1, 0);
        for (int ax2 = 0; ax2 < losses.dimen_list[2]; ax2++) {

          int index_offset = dl_dfilter.flat_from_indexes(x, 0, 0);
          float current_element_in_loss_gradient =
              losses.get_element_flat(current_start_of_row + ax2);

          loss_sum += current_element_in_loss_gradient;
          for (int f1 = 0; f1 < fil_size; f1++) {
            int affected_pixels_by_filter_row =
                last_input.flat_from_indexes(ax1 + f1, ax2);
            for (int f2 = 0; f2 < fil_size; f2++) {
              dl_dfilter.add_to_index(
                  index_offset + f1 * fil_size + f2,
                  current_element_in_loss_gradient *
                      last_input.get_element_flat(
                          affected_pixels_by_filter_row + f2));
            }
          }
        }
      }

      biases_data.data[x] -= learning_rate * loss_sum;
    }
    for (int el = 0; el < dl_dfilter.size; el++) {
      data.flat_assign(el, data.get_element_flat(el) -
                               learning_rate * dl_dfilter.get_element_flat(el));
    }
  }
};

// ---------------------------------------------------------------------------
// VERSION B -- the current version (layers.hpp today)
//
// Three changes, none of which touch the math:
//   1. F is a local pinned to 3 with a throw, so the compiler can PROVE the
//      trip count and unroll the 3x3 with f1 * F folded to literals
//   2. dl_dfilter is built from the shape instead of copy-then-zero --
//      vector<float>(n) is already zeroed, so the copy and the zeroing pass
//      both disappear
//   3. index_offset, the last_input strides, and the row base are hoisted out
//      of the loops that don't vary them
// ---------------------------------------------------------------------------
struct ConvBackpropCurrent {
  Tensor data;
  Tensor biases_data;
  int nodes;
  int fil_size = 3;
  Tensor last_input;
  float learning_rate = .01f;

  ConvBackpropCurrent(int neurons) : data(neurons, 3, 3), biases_data(neurons) {
    this->nodes = neurons;
  }

  void backprop(Tensor &losses) {
    const int F = this->fil_size;
    if (F != 3) throw std::runtime_error("filter must be 3x3");

    Tensor dl_dfilter(data.dimen_list);

    for (int x = 0; x < nodes; x++) {
      float loss_sum = 0.0f;
      const int last_input_width = last_input.indexing_list[0];
      const int last_input_height = last_input.indexing_list[1];
      int index_offset = dl_dfilter.flat_from_indexes(x, 0, 0);

      for (int ax1 = 0; ax1 < losses.dimen_list[1]; ax1++) {
        int current_start_of_row = losses.flat_from_indexes(x, ax1, 0);

        for (int ax2 = 0; ax2 < losses.dimen_list[2]; ax2++) {
          float current_element_in_loss_gradient =
              losses.get_element_flat(current_start_of_row + ax2);

          loss_sum += current_element_in_loss_gradient;

          int starting_index_of_pixels_affected_by_filter =
              last_input.flat_from_indexes(ax1, ax2);
          for (int f1 = 0; f1 < F; f1++) {
            for (int f2 = 0; f2 < F; f2++) {
              dl_dfilter.add_to_index(
                  index_offset + f1 * F + f2,
                  current_element_in_loss_gradient *
                      last_input.get_element_flat(
                          starting_index_of_pixels_affected_by_filter +
                          f1 * last_input_width + f2));
            }
          }
        }
      }

      biases_data.data[x] -= learning_rate * loss_sum;
    }
    for (int el = 0; el < dl_dfilter.size; el++) {
      data.flat_assign(el, data.get_element_flat(el) -
                               learning_rate * dl_dfilter.get_element_flat(el));
    }
  }
};

template <class Conv> Conv make_layer(int neurons, Tensor &image) {
  lcg_state = 12345u;
  Conv layer(neurons);
  for (int i = 0; i < layer.data.size; i++) layer.data.data[i] = next_val();
  for (int i = 0; i < layer.biases_data.size; i++)
    layer.biases_data.data[i] = next_val();
  layer.last_input = image;
  return layer;
}

template <class Conv>
double time_version(const char *name, Tensor &image, Tensor &grad,
                    double *checksum) {
  const int kIters = 2000;
  Conv layer = make_layer<Conv>(20, image);

  // one call for the correctness check, on a layer nothing has touched yet
  Conv fresh = make_layer<Conv>(20, image);
  fresh.backprop(grad);
  double sum = 0.0;
  for (int i = 0; i < fresh.data.size; i++) sum += fresh.data.data[i];
  for (int i = 0; i < fresh.biases_data.size; i++) sum += fresh.biases_data.data[i];
  *checksum = sum;

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIters; i++) layer.backprop(grad);
  auto t1 = std::chrono::steady_clock::now();

  double us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
  printf("  %-38s %8.2f us/image\n", name, us);
  return us;
}

int main() {
  lcg_state = 999u;
  Tensor image(28, 28);
  for (int i = 0; i < image.size; i++) image.data[i] = next_val();

  Tensor grad(20, 26, 26);
  for (int i = 0; i < grad.size; i++) grad.data[i] = next_val() * 0.001f;

  double checksum_a = 0.0, checksum_b = 0.0;

  // first pass runs cold (cache + CPU frequency ramp), read the second block
  for (int pass = 0; pass < 2; pass++) {
    printf("pass %d%s\n", pass + 1, pass == 0 ? " (cold -- ignore)" : "");
    double a = time_version<ConvBackpropOld>(
        "A: old (runtime bound, copy + zero)", image, grad, &checksum_a);
    double b = time_version<ConvBackpropCurrent>(
        "B: current (pinned + lifted)", image, grad, &checksum_b);
    printf("  -> current is %.2fx faster\n\n", a / b);
  }

  double diff = checksum_a - checksum_b;
  if (diff < 0) diff = -diff;
  printf("weights+biases checksum  A=%.8f  B=%.8f\n", checksum_a, checksum_b);
  bool ok = diff < 1e-5;
  printf("  diff=%g -> %s\n", diff, ok ? "MATCH" : "MISMATCH");
  return ok ? 0 : 1;
}
