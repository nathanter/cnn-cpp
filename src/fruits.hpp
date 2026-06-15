
#include <filesystem>
#include <vector>
using namespace std;

#ifndef LOADING_FRUITS
#define LOADING_FRUITS

vector<vector<std::filesystem::path>>
load_fruits_dataset(std::filesystem::path training);

#endif
