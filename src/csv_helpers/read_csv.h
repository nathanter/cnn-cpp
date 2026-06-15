#ifndef READ_CSV_H
#define READ_CSV_H

#include <filesystem>
#include <vector>

#include "../tensor.hpp"

std::vector<std::vector<float>>
read_csv_rows(const std::filesystem::path &path, bool skip_header = false);

Tensor read_tensor_from_csv(const std::filesystem::path &path,
                            const std::vector<int> &dimensions);

#endif
