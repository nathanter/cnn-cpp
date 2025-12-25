// code that was previously used for loading fruits dataset

#include "fruits.h"
#include "tensor.h"
#include <filesystem>

using namespace std;

// takes path to training data for fruits dataset
// returns a vector of vector of paths.
// each inner vector is a folder and the folder names
vector<vector<std::filesystem::path>>
load_fruits_dataset(std::filesystem::path training) {

  string p = "";
  vector<vector<std::filesystem::path>> folder_names;
  int folder_index = 0;

  for (auto const &dir_entry : std::filesystem::directory_iterator{training}) {
    folder_names.push_back(vector<std::filesystem::path>());
    if (!dir_entry.is_directory()) {
      cout << "failure" << endl;
      continue;
    }
    for (auto const &img_entry :
         std::filesystem::directory_iterator{dir_entry}) {

      folder_names[folder_index].push_back(img_entry.path());
    }
    folder_index++;
  }
  return folder_names;
}
