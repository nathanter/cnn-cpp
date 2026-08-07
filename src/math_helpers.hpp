
#ifndef MATH_HELPERS_H
#define MATH_HELPERS_H
#include "tensor.hpp"

Tensor apply_relu_back_prop(Tensor &valid_spots, Tensor &losses);

Tensor apply_relu_activation(Tensor &input);

void apply_soft_max(Tensor &input);

Tensor soft_max_opertation(Tensor &input, int correct_label);

int get_max(const Tensor& input);

float convert_to_float(unsigned char r, unsigned char g, unsigned char b);

#endif
