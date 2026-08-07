#include <algorithm>
#include <cmath>
#include <random>
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
  void update_learning_rate(float new_rate) {
    (*this).learning_rate = new_rate;
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

  // note this function is moving towards just doing the arithmetic by hand because using the variable arg functions in tensor is too expensive.
  Tensor feed_forward(Tensor &input) {
    this->last_input = input;
    const int F = this -> fil_size;
    if (F != 3) throw std::runtime_error("filter must be 3x3"); 
    // If make this function a template function, add general dispatch function that dispatches different versions of the template function based on template size.

    // not any tensor safe
    //  output tensor is a convoluted array representing the convoluted image
    //  for each node
    Tensor output(this->nodes, input.dimen_list[0] - 2,
                  input.dimen_list[1] - 2);


    
    // for x indexes
    // run 3 by 3 filter on 0
    // all the way to size - 3
    int out_index = 0;
    for (int node = 0; node < this->nodes; node++) {
      int start_of_filter_in_conv_weights = this -> data.flat_from_indexes(node,0,0);
      // this tracks the same thing as input.flat_from_indexes(ax1,ax2) by incrementing it every inner loop of the axis loop
    
      for (int ax1 = 0; ax1 <= input.dimen_list[0] - 3; ax1++) {
        for (int ax2 = 0; ax2 <= input.dimen_list[1] - 3; ax2++) {
          float sum = 0.00;
          int start_of_filter_in_input = ax1 * input.indexing_list[0] + ax2 * input.indexing_list[1];
            
          // using F here (the const we established above) allows the compiler to stop looping in the machine instructions which optimizes this portion marginally
          for (int f1 = 0; f1 < F; f1++) {
            int place_in_input_tensor = input.indexing_list[0] * f1 + start_of_filter_in_input;
            for (int f2 = 0; f2 < F; f2++) {
              sum += static_cast<float>(
                  input.get_element_flat(place_in_input_tensor + f2) *
                  this -> data.get_element_flat(start_of_filter_in_conv_weights + f1 * F + f2));
            }
          }

          sum += biases_data.get_element_flat(node);

          output.flat_assign(out_index,sum);
          out_index++;
        }

      }
    }
    return output;
  }

  
  void backprop(Tensor &losses) {
    const int F = this -> fil_size;
    if (F != 3) throw std::runtime_error("filter must be 3x3");
   // same reasoning as feed forward. allows unrolling.

    Tensor dl_dfilter(data.dimen_list);

    for (int x = 0; x < nodes; x++) {
      // nodes is number of filters here
      // f1 and f2 are traversing the dimensions of the filter
      float loss_sum = 0.0f;
      // loss_sum is the sum of all gradients for this filter, will be
      // subtracted from bias.
      const int last_input_width = last_input.indexing_list[0];
      const int last_input_height = last_input.indexing_list[1];
      int index_offset = dl_dfilter.flat_from_indexes(x, 0, 0);
      // index_offset is the location where the current filter stars in the
      // conv layers data
      for (int ax1 = 0; ax1 < losses.dimen_list[1]; ax1++) {
        int current_start_of_row = losses.flat_from_indexes(x,ax1,0);

        for (int ax2 = 0; ax2 < losses.dimen_list[2]; ax2++) {

          

          float current_element_in_loss_gradient =
            losses.get_element_flat(current_start_of_row + ax2);

          // current_element_in_loss_gradient is the element in the input
          // gradient that affects all the filters;

          loss_sum += current_element_in_loss_gradient;
          int starting_index_of_pixels_affected_by_filter = last_input.flat_from_indexes(ax1, ax2);
          for (int f1 = 0; f1 < F; f1++) {

            /*int affected_pixels_by_filter_row =
                last_input.flat_from_indexes(ax1 + f1, ax2);*/



            for (int f2 = 0; f2 < F; f2++) {
                // TODY: check if changes here are right
              dl_dfilter.add_to_index(
                  index_offset + f1 * F + f2,
                  current_element_in_loss_gradient *
                      last_input.get_element_flat(
                        starting_index_of_pixels_affected_by_filter + f1 * last_input_width + f2));
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
    // use HE intialization here as with above.
    float stddev = sqrt(2.0f / this->inputs);
    std::normal_distribution<float> dist(0.0f, stddev);
    static std::default_random_engine generator(std::random_device{}());

    for (int x = 0; x < this->nodes; x++) {
      for (int y = 0; y < this->inputs; y++) {
        weights.flat_assign(weights.flat_from_indexes(x, y), dist(generator));
      }
    }
  }
  void randomize_biases() {
    // small random instead of 0 because relu.
    for (int x = 0; x < this->nodes; x++) {

      biases.flat_assign(x, 0.01f);
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


  void update_learning_rate(float new_rate) {
    (*this).learning_rate = new_rate;
  }

  Tensor back_prop(Tensor &losses) {

    Tensor newloss(last_inputs.dimen_list);

    newloss.zeros();

    // we update loss by calculating the d_loss/d_weight or d_loss/d_bias.
    // the relu layer gets the delta loss with respect to the output of this function including the activation layer.
    // the result of relu backprop returns the delta_loss/delta_z (z is output of the weight sum)
    // delta_loss/delta_z * delta_z/delta_w = delta_loss/d_weight
    // d_z/d_w = input  because z = w1*x1 + w2*x2 + whatever.
    // everything else is constants that go to 0. 

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

        // assign new loss before changing weights
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
          const int pool_start_x = x * 2;
          const int pool_start_y = y * 2;
          const int pool_end_x =
              std::min(pool_start_x + 2, input.dimen_list[1]);
          const int pool_end_y =
              std::min(pool_start_y + 2, input.dimen_list[2]);

          float pooled_value = -INFINITY;
          int pooled_origin = offset_1;

          for (int source_x = pool_start_x; source_x < pool_end_x; source_x++) {
            for (int source_y = pool_start_y; source_y < pool_end_y; source_y++) {
              const int source_index =
                  input.flat_from_indexes(neuron, source_x, source_y);
              if (i_d[source_index] > pooled_value) {
                pooled_value = i_d[source_index];
                pooled_origin = source_index;
              }
            }
          }

          max_origin[last_output.flat_from_indexes(neuron, x, y)] =
              pooled_origin;

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

    new_dimenx = losses.dimen_list[1];
    new_dimeny = losses.dimen_list[2];

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
