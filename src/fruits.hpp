
#include <filesystem>
#include <vector>


#ifndef LOADING_FRUITS
#define LOADING_FRUITS
using std::vector;

vector<vector<std::filesystem::path>>
load_fruits_dataset(std::filesystem::path training);

#endif
