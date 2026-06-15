#ifndef IMAGE_TO_CSV_H
#define IMAGE_TO_CSV_H

#include <string>
#include "../tensor.hpp"

Tensor image_file_to_tensor(const std::string &path, Tensor &image);

#endif
