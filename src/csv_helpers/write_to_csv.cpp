#include "write_to_csv.h"

#include <fstream>
#include <stdexcept>

using namespace std;

void write_tensor_to_csv(const Tensor &tensor, const filesystem::path &path) {
  ofstream output(path);
  if (!output.is_open()) {
    throw runtime_error("Unable to open output file: " + path.string());
  }

  for (int index = 0; index < tensor.size; index++) {
    if (index != 0) {
      output << ",";
    }
    output << tensor.get_element_flat(index);
  }
  output << '\n';
}
