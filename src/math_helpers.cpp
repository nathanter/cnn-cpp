#include "math_helpers.hpp"
#include "tensor.hpp"

using namespace std;

Tensor apply_relu_back_prop(Tensor &valid_spots, Tensor &losses) {
  // this function is used in backpropgation.
  // zeros out parts of the gradient where the output of the conv layer was
  // originally negative valid spots is defined below, should be same size and
  // shape as losses

  vector<float> &v_d = valid_spots.data;
  vector<float> &l_d = losses.data;
  for (int x = 0; x < losses.size; x++) {

    l_d[x] *= v_d[x];
  }
  return losses;
}

Tensor apply_relu_activation(Tensor &input) {
  // this function applies relu to a tensor
  // edit input in place
  // remember where values are positive/negative
  // store this in mask

  Tensor mask = input;
  mask.zeros();

  for (int x = 0; x < input.size; x++) {
    if (input.data[x] > 0) {
      mask.data[x] = 1;
    } else {
      mask.data[x] = 0;
      input.flat_assign(x, 0);
    }
  }
  return mask;
}

float calc_exp_sum(Tensor &input, float maxv) {
  // this function calculates the sum of e^x for all elements x in a tensor
  // it substracts max value seen in the tensor to keep values in a manageable
  // range

  float sum = 0.0f;

  for (int x = 0; x < input.size; x++) {
    sum += exp(input.get_element_flat(x) - maxv);
  }
  return sum;
}

void apply_soft_max(Tensor &input) {
  // this function converts each element x in a tensor to the e^x

  float maxv = -INFINITY;
  for (int x = 0; x < input.size; x++) {
    maxv = max(maxv, input.get_element_flat(x));
  }
  float esum = calc_exp_sum(input, maxv);

  float target_val;
  for (int x = 0; x < input.size; x++) {
    target_val = (exp(input.get_element_flat(x) - maxv) / esum);
    input.flat_assign(x, target_val);
  }
}

Tensor soft_max_opertation(Tensor &input, int correct_label) {

  // function writes softmaxed inputs into place
  // returns a tensor of the derivative of resultant values relative to the
  // result totals of the last pass the tensor should be equal to soft_max
  // probabilities - expected value;

  Tensor loss(input.size);
  float maxv = -INFINITY;
  for (int x = 0; x < input.size; x++) {
    maxv = max(maxv, input.get_element_flat(x));
  }
  float esum = calc_exp_sum(input, maxv);

  float target_val;
  for (int x = 0; x < input.size; x++) {
    target_val = (exp(input.get_element_flat(x) - maxv) / esum);
    input.flat_assign(x, target_val);
    if (x == correct_label) {
      loss.flat_assign(x, target_val - 1);
    } else {
      loss.flat_assign(x, target_val - 0);
    }
  }
  return loss;
}

int get_max(Tensor input) {
  // returns the index of the max element in a tensor

  float max = -INFINITY;
  int index = 0;
  for (int x = 0; x < input.size; x++) {
    if (input.get_element_flat(x) > max) {
      max = input.get_element_flat(x);
      index = x;
    }
  }
  return index;
}

float convert_to_float(unsigned char r, unsigned char g, unsigned char b) {
  // used for converting images withs stb image to tensor with no channels
  // equation sourced should translate values to grayscale

  float intval = (0.2989f * r + 0.5870f * g + 0.1140f * b) / 255.0f;
  return intval;
}