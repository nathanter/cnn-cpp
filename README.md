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



# Results:

Note: The overall accuracy for the fruits_360 dataset is not as high as normal image classification should be. This is because the fruit images require 3d convolution to consider the multiple channels of colors, but this implementation does not have 3d convolution. However, the results were still taken and considered to compare with the other dataset.


Fashion-Mnist- 10 Classes:
* Trial 1: 90.04.  20 epochs. 10 convolution layer nodes
* Trial 2: 91.29.  15 epochs. 20 convolution layer nodes.
* Trial 3: 80.23.  5 epochs. only 1 layer.

Fruits- 6 Classes(first 6 folders of apples out of 237 fruits):
* Trials 1-3: ~30%

