#include "image_to_csv.h"

#include <stdexcept>

#include "../math_helpers.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Tensor image_file_to_tensor(const std::string &path, Tensor &image) {
  int width = 0;
  int height = 0;
  int channels = 0;

  unsigned char *img = stbi_load(path.c_str(), &width, &height, &channels, 0);

  if (img == nullptr) {
    throw std::runtime_error("Failed to load image: " + path);
  }

  if (channels < 1) {
    stbi_image_free(img);
    throw std::runtime_error("Image has no channels: " + path);
  }

  if (image.dimen_list.size() != 2 || image.dimen_list[0] != height ||
      image.dimen_list[1] != width) {
    stbi_image_free(img);
    throw std::runtime_error("Image dimensions do not match tensor dimensions");
  }

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const int index = channels * (y * width + x);
      const unsigned char r = img[index];
      const unsigned char g = channels > 1 ? img[index + 1] : r;
      const unsigned char b = channels > 2 ? img[index + 2] : r;
      image.flat_assign(image.flat_from_indexes(y, x), convert_to_float(r, g, b));
    }
  }

  stbi_image_free(img);
  return image;
}
