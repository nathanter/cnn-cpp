To build and run.

```sh
g++ -std=c++17 -O3 main.cpp -o main && ./main
```

Implementation of a convolutional neural network from scratch.

Tested by running on subset of fruits 360 dataset and Mnist-fashion.

https://www.kaggle.com/datasets/zalando-research/fashionmnist

https://www.kaggle.com/datasets/moltean/fruits 


The neural network consists of a convolutional layer that uses 3x3 filters. The activation function for this layer is RELU.
Following the convolutional layer is a pooling layer that shortens the input using MaxPool on a 2x2 area
Finally a fully connected dense layer with the same amount of neurons as inputs(6 for tested subsection of fruits 360 and 10 for mnist-fashion)

along with the implemented neural network is classes for a Tensor object, a 3x3 2D Conv Layer, a 2x2 2D Pool layer, and a layer with weights and biases.

# Results:

Note: The overall accuracy for the fruits_360 dataset is not as high as normal image classification should be. This is because the fruit images require 3d convolution to consider the multiple channels of colors, but this implementation does not have 3d convolution. However, the results were still taken and considered to compare with the other dataset.


Fashion-Mnist- 10 Classes:
* Trial 1: 77.7%
* Trial 2: 81.59%
* Trial 3: 80.23

Fruits- 6 Classes(first 6 folders of apples out of 237 fruits):
* Trials 1-3: ~30%


