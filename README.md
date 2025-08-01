# image_classifier_CNN

This ML project aims to build an image classifier using CNN (Convolutional Neural Network) I used homemade CNN and Google Inception.

# Data

The dataset used for this project contains 28000 medium quality images belonging to 10 categories: dog, cat , horse, spyder, butterfly, chicken, sheep, cow, squirrel and elephant. All this images have been collected from "google images" and gave been checked by human.

# Architecture

- INSERT THE src/cnn_architecture.png

- Input Layer: Image RGB with 224x224x3
- Convolutional Block 1: Convolution 2D (32 filters 3x3, act. funct. ReLU) + Max Pooling (2x2)
- Convolutional Block 2: Convolution 2D (64 filters 3x3, act. funct. ReLU) + Max Pooling (2x2)
- Convolutional Block 3: Convolution 2D (128 filters 3x3, act. funct. ReLU) + Maz Pooling (2x2)
- Convolutional Block 4: Convolution 2D (256 filters 3x3, act. funct. ReLu) + Max Pooling (2x2)
- 1D Vector containg all the conv. 
- Dense Layer: Learning the features combinations (512 neurons, act. funct. ReLU)
- Dropout: 50% of the neurons (prevents overfitting)
- Output: Number of Classes (act. funct. Softmax)