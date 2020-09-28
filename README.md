# Telugu-Character-Recognition-using-CNN

Convolutional Neural Networks (ConvNets or CNNs) are a category of Neural Networks that have proven very effective in areas such as image recognition and classification. 


There are four main operations in the ConvNet:

    Convolution
    Non Linearity (ReLU)
    Pooling or Sub Sampling
    Classification (Fully Connected Layer)

These operations are the basic building blocks of every Convolutional Neural Network.

The Convolution Step:

ConvNets derive their name from the “convolution” operator. The primary purpose of Convolution in case of a ConvNet is to extract features from the input image. Convolution preserves the spatial relationship between pixels by learning image features using small squares of input data. 


Non Linearity (ReLU):

ReLU is an element wise operation (applied per pixel) and replaces all negative pixel values in the feature map by zero. The purpose of ReLU is to introduce non-linearity in our ConvNet, since most of the real-world data we would want our ConvNet to learn would be non-linear (Convolution is a linear operation – element wise matrix multiplication and addition, so we account for non-linearity by introducing a non-linear function like ReLU).

The Pooling Step:

Spatial Pooling (also called subsampling or downsampling) reduces the dimensionality of each feature map but retains the most important information. Spatial Pooling can be of different types: Max, Average, Sum etc.

Fully Connected Layer:

The Fully Connected layer is a traditional Multi Layer Perceptron that uses a softmax activation function in the output layer (softmax classifier is used here). The term “Fully Connected” implies that every neuron in the previous layer is connected to every neuron on the next layer. 
The output from the convolutional and pooling layers represent high-level features of the input image. The purpose of the Fully Connected layer is to use these features for classifying the input image into various classes based on the training dataset.
