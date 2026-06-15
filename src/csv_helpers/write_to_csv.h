#ifndef WRITE_TO_CSV_H
#define WRITE_TO_CSV_H

#include <filesystem>

#include "../tensor.hpp"

void write_tensor_to_csv(const Tensor &tensor, const std::filesystem::path &path);

#endif
