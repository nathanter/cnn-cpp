#include "layers.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include "math_helpers.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include <chrono>
#include <sstream>

using namespace std;

vector<vector<float>> translate_csv_to_tensor(std::ifstream &input) {

  // reads a csv and returns each row as vector of float in side another vector

  vector<vector<float>> csv_data;
  std::string line;

  while (std::getline(input, line)) {
    std::stringstream ss(line);
    std::string cell;
    std::vector<float> row;

    while (std::getline(ss, cell, ',')) {
      row.push_back(std::stof(cell));
    }

    csv_data.push_back(row);
  }
  return csv_data;
}

Tensor csv_line_to_tensor(vector<float> csv_line, int &label, Tensor &output) {

  // takes one vector of float values to a tensor normalized to 0-1 by dividing
  // by 255.0f

  label = csv_line[0];

  vector<float> &o_d = output.data;
  for (int pix = 1; pix < csv_line.size(); pix++) {
    o_d[pix - 1] = csv_line[pix] / 255.0f;
  }

  return output;
}

Tensor translate_image_to_tensor(string path, Tensor &image) {
  // takes an image loaded by stb_image and translates it to tensor

  int width, height, channels;

  unsigned char *img = stbi_load(path.c_str(), &width, &height, &channels, 0);

  if (img == nullptr) {
    std::cout << "Failed to load image\n";
  }

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int index = channels * (y * width + x);
      unsigned char r = img[index];
      unsigned char g = img[index + 1];
      unsigned char b = img[index + 2];
      image.flat_assign(image.flat_from_indexes(x, y),
                        convert_to_float(r, g, b));
    }
  }
  stbi_image_free(img);
  return image;
}

int main(int argc, char* argv[]) {

  //params that need to be set before running
  const int WIDTH = 28;
  const int HEIGHT = 28;
  int labels = 10;
  // preinitialization 
  Conv_Layer2D cvl1(10);
  Pool_Layer2x2 pl;
  Layer l1(labels, ((WIDTH - 2) / 2) * ((WIDTH - 2) / 2) * 10);
  
	

  // Parse input.
  if (argc < 2) {
    printf("Please supply a command [train, eval].\n");
  }

  std::string command_string = std::string(argv[1]);

  // Train:
  // - take in path
  // if folder of images: convert to csv
  // else. take csv
  // run through train function
  // return and print path to text file that stores weights in csv
  if (command_string == "train") {

    

  // eval:
  // - take in path
  // if image: convert to csv
  // else. take csv
  // use model path to load weights
  // evaluate and return prediction
  // return and print path to text file that stores weights in csv
  } else if (command_string == "eval") {

  } else {
    printf("Provided command \"%s\" is invalid, please provide a valid command [train, eval].", command_string.c_str());
  }

  for (uint i = 0; i < argc; i += 1) {
    printf("%s ", argv[i]);
  }


  cvl1.Rand_filter();
  l1.randomize_weights();
  l1.randomize_biases();

  //const std::filesystem::path training{"datasets/fruits-360_100x100/fruits-360/Training"};

  Tensor image(WIDTH, HEIGHT); // preinitializing image
  Tensor result(1, 1);         // preinitializing result tensor

  string p = "";
  vector<vector<std::filesystem::path>> folder_names;
  // int folder_index = 0;

  // load csv dataset;
  // load rows into csv data, index 0 is label rest are inputs
  // initialize training data

  ifstream input("datasets/Mnist-fashion/fashion-mnist_train.csv");
  string trash;
  getline(input, trash);
  vector<vector<float>> csv_data = translate_csv_to_tensor(input);

  // initialize test data
  ifstream test_input("datasets/Mnist-fashion/fashion-mnist_test.csv");
  getline(test_input, trash);
  vector<vector<float>> csv_test_data = translate_csv_to_tensor(test_input);

  int label = 0;
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  const int runs_in_epoch = 60000;
  // preinitializing string   p
  for (int epoch = 0; epoch < 3; epoch++) {
    cout << epoch << endl;
    result = Tensor(1, 1);

    for (auto runs = 0; runs < runs_in_epoch; runs++) {
      // this process is for loading from the fruits dataset
      // uncomment to use that dataset

      // this line is for loading fashion mnist
      // comment when unused
      image = csv_line_to_tensor(csv_data[runs], label, image);
      //

      result = cvl1.feed_forward(image);

      Tensor relu_back_helper = apply_relu_activation(result);

      result = pl.Pool(result);

      result = l1.feed_forward(result);

      Tensor loss = soft_max_opertation(result, label);

      loss = l1.back_prop(loss);
      loss = pl.back_prop(loss);
      loss = apply_relu_back_prop(relu_back_helper, loss);
      cvl1.backprop(loss);
    }
  }
  // resetting label and p
  // preinitializing correct counter for correct predictions and cases counter
  p = "";
  label = 0;
  int correct = 0;
  int cases = 0;

  cases = csv_test_data.size();
  // testing for fruits
  /*
  const std::filesystem::path
  testing{"datasets/fruits-360_100x100/fruits-360/Test"}; for (auto const
  &dir_entry : std::filesystem::directory_iterator{testing})
  {
      cout << label << endl;

      std::filesystem::path subdir{dir_entry.path()};
      if (!dir_entry.is_directory())
      {
          cout << "failure" << endl;
          continue;
      }
      for (auto const &img_entry : std::filesystem::directory_iterator{subdir})
      {
          cases++;

          p = img_entry.path().string();
          image = translate_image_to_tensor(p,image);

          result = cvl1.feed_forward(image);
          Tensor relu_back_helper = apply_RELU_activation(result);
          result = pl.Pool(result);
          result = l1.feed_forward(result);
          apply_soft_max(result);
          if (get_max(result) == label)
          {
              correct++;
          }
      }
      label++;
  }

  */

  // csv_test_data is all the input images in the test case;
  cases = csv_test_data.size();

  // preinitialize testing label
  int testing_label;

  for (int entry = 0; entry < csv_test_data.size(); entry++) {
    image = csv_line_to_tensor(csv_test_data[entry], testing_label, image);

    result = cvl1.feed_forward(image);
    Tensor relu_back_helper = apply_relu_activation(result);
    result = pl.Pool(result);

    result = l1.feed_forward(result);

    Tensor loss = soft_max_opertation(result, testing_label);

    if (testing_label == get_max(result)) {
      correct++;
    }
  }

  cout << "accuracy :" << 100.0f * correct / cases << endl;
}
