#include "tensor.hpp"

#ifndef CONV_H
#define CONV_H

class Conv_Layer2D {

  // layer to apply 3x3 convolution filter to image represented by Tensor class.
public:
  Tensor data;
  Tensor biases_data;
  int nodes;
  int fil_size = 3;
  Tensor last_input;
  float learning_rate = .01f;
  Conv_Layer2D(int neurons) : data(neurons, 3, 3), biases_data(neurons) {
    this->nodes = neurons;
  }
  void Rand_filter() {
    int fan_in = 9;
    float stddev = sqrt(2.0f / fan_in);
    std::normal_distribution<float> dist(0.0f, stddev);
    static std::default_random_engine generator(std::random_device{}());

    for (int x = 0; x < this->nodes; x++) {
      for (int y = 0; y < 3; y++) {
        for (int z = 0; z < 3; z++) {
          data.flat_assign(data.flat_from_indexes(x, y, z),
                           (dist(generator)) + .01f);
        }
      }
    }
    for (int x = 0; x < this->nodes; x++) {

      this->biases_data.flat_assign(x, 0.01f);
    }
  }
  void print_filters() {
    // debugging method
    data.print_data();
  }
  Tensor feed_forward(Tensor &input) {
    this->last_input = input;
    // not any tensor safe
    //  output tensor is a convoluted array representing the convoluted image
    //  for each node
    Tensor output(this->nodes, input.dimen_list[0] - 2,
                  input.dimen_list[1] - 2);

    // for x indexes
    // run 3 by 3 filter on 0
    // all the way to size - 3

    for (int node = 0; node < this->nodes; node++) {

      for (int ax1 = 0; ax1 <= input.dimen_list[0] - 3; ax1++) {
        for (int ax2 = 0; ax2 <= input.dimen_list[1] - 3; ax2++) {
          float sum = 0.00;
          for (int f1 = 0; f1 < fil_size; f1++) {
            for (int f2 = 0; f2 < fil_size; f2++) {
              sum += static_cast<float>(
                  input.get_element_by_indexes(ax1 + f1, ax2 + f2) *
                  this->data.get_element_by_indexes(node, f1, f2));
            }
          }
          sum += biases_data.get_element_flat(nodes);

          output.flat_assign(output.flat_from_indexes(node, ax1, ax2), sum);
        }
      }
    }
    return output;
  }

  void backprop(Tensor &losses) {
    Tensor dl_dfilter = data;
    dl_dfilter.zeros();

    for (int x = 0; x < nodes; x++) {
      // nodes is number of filters here
      // f1 and f2 are traversing the dimensions of the filter
      float loss_sum = 0.0f;
      // loss_sum is the sum of all gradients for this filter, will be
      // subtracted from bias.
      for (int ax1 = 0; ax1 < losses.dimen_list[1]; ax1++) {
        int current_start_of_row = losses.flat_from_indexes(x,ax1,0);
        for (int ax2 = 0; ax2 < losses.dimen_list[2]; ax2++) {

          int index_offset = dl_dfilter.flat_from_indexes(x, 0, 0);
          // index_offset is the location where the current filter stars in the
          // conv layers data
          float current_element_in_loss_gradient =
            losses.get_element_flat(current_start_of_row + ax2);

          // current_element_in_loss_gradient is the element in the input
          // gradient that affects all the filters;
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
#endif

#ifndef WEIGHTED_LAYER
#define WEIGHTED_LAYER
class Layer {
public:
  Tensor weights;
  Tensor biases;
  int nodes;
  int inputs;
  Tensor last_inputs;
  float learning_rate = .01f;

  Layer(int neurons, int inputsnum)
      : weights(neurons, inputsnum), biases(neurons) {
    this->nodes = neurons;
    this->inputs = inputsnum;
  }

  void randomize_weights() {

    for (int x = 0; x < this->nodes; x++) {
      for (int y = 0; y < this->inputs; y++) {
        weights.flat_assign(weights.flat_from_indexes(x, y),
                            (static_cast<float>(rand() % 200) - 100.00) /
                                100.00);
      }
    }
  }
  void randomize_biases() {

    for (int x = 0; x < this->nodes; x++) {

      biases.flat_assign(x,
                         (static_cast<float>(rand() % 200) - 100.00) / 100.00);
    }
  }

  Tensor feed_forward(Tensor &input_tensor) {

    if (input_tensor.size != this->inputs) {
      throw std::runtime_error("Number of indexes does not match dimensions");
    }

    this->last_inputs = input_tensor;

    Tensor result(nodes);
    auto &i_d = input_tensor.data;
    auto &w_d = this->weights.data;

    for (int x = 0; x < nodes; x++) {
      float d_product = 0;
      int index_offset = x * this->inputs;
      for (int y = 0; y < this->inputs; y++) {
        d_product += i_d[y] * w_d[index_offset + y];
      }
      d_product += biases.get_element_flat(x);
      result.flat_assign(x, d_product);
    }
    return result;
  }

  void print_filters() {
    // debugging method
    weights.print_data();
  }

  Tensor back_prop(Tensor &losses) {

    Tensor newloss(last_inputs.dimen_list);

    newloss.zeros();

    auto &w_d = weights.data;
    auto &b_d = biases.data;
    auto &l_d = losses.data;
    auto &n_d = newloss.data;
    for (int x = 0; x < nodes; x++) {
      float loss_for_this_label = l_d[x];
      b_d[x] -= learning_rate * loss_for_this_label;

      int current_index_offset = x * inputs;
      for (int y = 0; y < inputs; y++) {

        n_d[y] += w_d[current_index_offset + y] * loss_for_this_label;

        // assign n  ew loss before changing weights
        w_d[current_index_offset + y] -= learning_rate *
                                         last_inputs.get_element_flat(y) *
                                         loss_for_this_label;
      }
    }

    return newloss;
  }
};
#endif

#ifndef POOL_LAYER
#define POOL_LAYER

class Pool_Layer2x2 {
public:
  vector<int> last_input_dimensions;
  vector<int> max_origin;
  Pool_Layer2x2() {}
  Tensor Pool(Tensor &input) {
    // assume three dimensional array
    // first index is filter that produced result
    // second dimensions

    // look at each 2 x 2 region and find max
    this->last_input_dimensions = input.dimen_list;
    int new_dimenx;
    int new_dimeny;

    if (input.dimen_list[1] % 2 == 1) {
      new_dimenx = input.dimen_list[1] / 2 + 1;
      new_dimeny = input.dimen_list[2] / 2 + 1;
    } else {
      new_dimenx = input.dimen_list[1] / 2;
      new_dimeny = input.dimen_list[2] / 2;
    }

    Tensor last_output = Tensor(input.dimen_list[0], new_dimenx, new_dimeny);

    last_output.zeros();
    this->max_origin = vector<int>(last_output.size);
    // fix this

    vector<float> &i_d = input.data;
    for (int neuron = 0; neuron < input.dimen_list[0]; neuron++) {

      for (int x = 0; x < new_dimenx; x++) {
        for (int y = 0; y < new_dimeny; y++) {
          int offset_1 = input.flat_from_indexes(neuron, x * 2, y * 2);
          // offset for first row of 2x2
          int offset_2 = input.flat_from_indexes(neuron, x * 2 + 1, y * 2);
          // offset for second layer of 2x2
          float pooled_value = max({i_d[offset_1], i_d[offset_2],
                                    i_d[offset_1 + 1], i_d[offset_2 + 1]});

          for (int pool_area_x = 0; pool_area_x < 2; pool_area_x++) {

            if (i_d[offset_1 + pool_area_x] == pooled_value) {
              max_origin[last_output.flat_from_indexes(neuron, x, y)] =
                  offset_1 + pool_area_x;
              break;
            } else if (i_d[offset_2 + pool_area_x] == pooled_value) {
              max_origin[last_output.flat_from_indexes(neuron, x, y)] =
                  offset_2 + pool_area_x;
              break;
            }
          }

          last_output.flat_assign(last_output.flat_from_indexes(neuron, x, y),
                                  pooled_value);
        }
      }
    }
    return last_output;
  }
  Tensor back_prop(Tensor &losses) {

    Tensor dlwrti(last_input_dimensions);
    // delta loss with respect to inputs
    dlwrti.zeros();
    int new_dimenx;
    int new_dimeny;

    vector<int> &m_d = max_origin;

    new_dimenx = dlwrti.dimen_list[1] / 2;
    new_dimeny = dlwrti.dimen_list[2] / 2;

    for (int n = 0; n < dlwrti.dimen_list[0]; n++) {
      for (int x = 0; x < new_dimenx; x++) {

        int index_offset_for_losses = losses.flat_from_indexes(n, x, 0);
        for (int y = 0; y < new_dimeny; y++) {
          dlwrti.data[m_d[index_offset_for_losses + y]] +=
              losses.get_element_flat(index_offset_for_losses + y);
        }
      }
    }
    return dlwrti;
  }
};

#endif