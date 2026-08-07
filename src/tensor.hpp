#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#ifndef TENSOR_H

#define TENSOR_H

using namespace std;

class Tensor {
private:
  int get_size() { return this->size; }

public:
  vector<float> data;
  // flat vector containing all data
  int dimensions;
  // how many dimensions there are

  vector<int> dimen_list;
  // list of size dimensions that stores each dimension size
  int size;
  // all dimensions multiplied together

  vector<int> indexing_list;
  // each index of this list is the product of all remainign indexes

  template <class... Types> float get_element_by_indexes(Types... indexes) const {
    return data[flat_from_indexes(indexes...)];
  }

  template <class... Types> int flat_from_indexes(Types... indexes) const {

    vector<int> arr{indexes...};
    if (static_cast<int>(arr.size()) != dimensions) {
      throw std::runtime_error("Number of indexes does not match dimensions");
    }

    int final_index = 0;
    for (int x = 0; x < dimensions; x++) {
      final_index += arr[x] * indexing_list[x];
    }
    return final_index;
  }

  void flat_assign(int flat, float val) { this->data[flat] = val; }
  
  float get_element_flat(int flat_index) const { return (this->data)[flat_index]; }

  void zeros() {
    for (int m = 0; m < (this->size); m++) {
      data[m] = 0;
    }
  }
  void randomize() {
    for (int m = 0; m < (this->size); m++) {
      data[m] = static_cast<float>(rand() % 100) / 100.0f;
    }
  }

  void ones() {
    for (int m = 0; m < (this->size); m++) {
      data[m] = 1;
    }
  }
  void print_data() {
    // debugging method
    for (int m = 0; m < (this->size); m++) {
      cout << data[m] << endl;
    }
  }
  void print_ilist() {
    for (int m = 0; m < (this->dimensions); m++) {
      cout << indexing_list[m] << endl;
    }
  }
  void add_to_index(int x, float v) { data[x] += v; }
  Tensor()
      : data(1, 0.0f), dimensions{1}, dimen_list(1, 1), size{1},
        indexing_list(1, 1) {}
  template <class... Types>
  Tensor(Types... args) : dimensions{sizeof...(args)} {

    vector<int> arr{args...};
    this->dimen_list = arr;
    this->indexing_list = vector<int>(dimensions);
    int total_size{1};

    for (int l = 0; l < this->dimensions; l++) {

      total_size *= arr[l];
    }

    this->indexing_list[dimensions - 1] = 1;
    int prod_so_far{1};
    for (int k = dimensions - 2; k >= 0; k -= 1) {
      prod_so_far *= dimen_list[k + 1];
      this->indexing_list[k] = prod_so_far;
    }

    this->data = vector<float>(total_size);
    this->size = total_size;
  }
  Tensor(const vector<int>& inputarray)
      : dimensions{static_cast<int>(inputarray.size())} {
    this->dimen_list = inputarray;
    int total_size{1};
    this->indexing_list = vector<int>(dimensions);
    for (int l = 0; l < static_cast<int>(inputarray.size()); l++) {

      total_size *= inputarray[l];
    }
    this->indexing_list[dimensions - 1] = 1;
    int prod_so_far{1};
    for (int k = dimensions - 2; k >= 0; k -= 1) {
      prod_so_far *= dimen_list[k + 1];
      this->indexing_list[k] = prod_so_far;
    }

    this->data = vector<float>(total_size);
    this->size = total_size;
  }
};

#endif
