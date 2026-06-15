#include "read_csv.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace std;

vector<vector<float>> read_csv_rows(const filesystem::path &path,
                                    bool skip_header) {
  ifstream input(path);
  if (!input.is_open()) {
    throw runtime_error("Unable to open csv file: " + path.string());
  }

  vector<vector<float>> csv_data;
  string line;

  if (skip_header) {
    getline(input, line);
  }

  while (getline(input, line)) {
    if (line.empty()) {
      continue;
    }

    stringstream ss(line);
    string cell;
    vector<float> row;

    while (getline(ss, cell, ',')) {
      row.push_back(stof(cell));
    }

    if (!row.empty()) {
      csv_data.push_back(row);
    }
  }

  return csv_data;
}

Tensor read_tensor_from_csv(const filesystem::path &path,
                            const vector<int> &dimensions) {
  const auto rows = read_csv_rows(path, false);
  Tensor tensor(dimensions);

  vector<float> flattened;
  for (const auto &row : rows) {
    flattened.insert(flattened.end(), row.begin(), row.end());
  }

  if (static_cast<int>(flattened.size()) != tensor.size) {
    throw runtime_error("Tensor file size does not match expected dimensions: " +
                        path.string());
  }

  for (int index = 0; index < tensor.size; index++) {
    tensor.flat_assign(index, flattened[index]);
  }

  return tensor;
}
