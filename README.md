## Convolutional Neural Network from scratch
Implementation of a convolutional neural network from scratch.

Tested by running on subset of fruits 360 dataset and Mnist-fashion.

https://www.kaggle.com/datasets/zalando-research/fashionmnist

https://www.kaggle.com/datasets/moltean/fruits 

## To build and run.

```sh
g++ -std=c++17 -O3 src/main.cpp src/math_helpers.cpp src/fruits.cpp src/csv_helpers/image_to_csv.cpp src/csv_helpers/read_csv.cpp src/csv_helpers/write_to_csv.cpp -o main
```

Train and save a model:

```sh
./main train
```
this saves 4 files to the default directory of models/fashion-mnist which will be the weights and biases for the conv and dense layers. 

train can be used along with a system path to store the models at a specific location
```sh
./main train {{path}}
```

```sh
./main train {{path}} x1 x2 x3
```
this can be used to make an arbitrary amount of connected dense layers with neuron size x. path indicates where to save the model. Must be used with path.

Evaluate a saved model on the default Fashion-MNIST test csv if models were intialized with no parameters to the train command.
```sh
./main eval
```

eval command has 3 inputs otherwise indicating the specific model_dir and input path to use and size of hidden layers. If there is a custom amount of dense layers (ie: not the default of only 1 output layer). Then the hidden layer sizes MUST be included. 
```sh
./main eval [model_dir] [input_path] [hidden_layer_sizes...]
```


# Architecture

The neural network consists of a convolutional layer that uses 3x3 filters. The activation function for this layer is RELU.
Following the convolutional layer is a pooling layer that shortens the input using MaxPool on a 2x2 area
Finally an arbitrary amount of fully connected dense layer with the same amount of neurons as labels(6 for tested subsection of fruits 360 and 10 for mnist-fashion). 

The results below were computed with either a hidden layer with 64 neurons and an output layer of 10 neurons or only one dense layer (the 10 neuron output layer) connected to the convolution layer.

along with the implemented neural network is classes for a Tensor object, a 3x3 2D Conv Layer, a 2x2 2D Pool layer, and a layer with weights and biases.


# optimization process:



Prior to Aug 5th the convolution layer took too long.

- Conv layer optimization for backpropagation and forward propagation included moving repeated calculations and vector initializations out of the main loop. 
- Removed unnecessary duplicated logic in get_element_by_indexes() that allows compiler to use flat_from_indexes() which was optimized by compiler instead of heap allocation. Get_element_by_indexed() build a new `vector<int>` which is allocated to heap. Size checks in flat_from_indexes() informs compiler on that sizing is correct so it can elide vector allocation and stop actually looping and use straight instructions.
- added similar safety/size checks to convolution backprop and other functions so it allows the compiler to also do allocation elision or loop unrolling (Only available when compiling with O3 and O2 tags) 
- Improved tensor intilization and indexing functions more efficient by using c style arrays instead of vector intilization. Improved convolution backprop from 80us to ~60 us
- Forward pass of convolution layer from > 6000 us to <20 us in total. Back propagation from >80us to ~25 us in latest changes.
- Back propagation used to take longer but did some optimizations before Aug 5th in earlier commits.
- This reduces 2 hours of training time to around a minute.


Measured with 20 filters, 28x28 input, 2000 iterations, median of 3
runs. Apple M1 (4 performance + 4 efficiency cores), 16 GB, macOS 15.6, Apple clang 17.0.0, arm64, `-O3` on build.

Files in tests/ have build commands for scripts that check if size checks actually cause speedups, check the total optimizations not including tensor changes, and measure current speed of convolution layer.


# Results:

Note: The overall accuracy for the fruits_360 dataset is not as high as normal image classification should be. This is because the fruit images require 3d convolution to consider the multiple channels of colors, but this implementation does not have 3d convolution. However, the results were still taken and considered to compare with the other dataset.


Fashion-Mnist- 10 Classes:
* Trial 1: 90.04.  20 epochs. 10 convolution layer nodes. Hidden layer size 64
* Trial 2: 91.77.  17 epochs. 32 convolution layer nodes. Hidden layer size 64
* Trial 3: 80.23.  5 epochs. only 1 layer.
* Trial 4: 92.23.  20 Epochs. 40 Convolution layer nodes. Hiddel layer size 64.

Fruits- 6 Classes(first 6 folders of apples out of 237 fruits):
* Trials 1-3: ~30%

