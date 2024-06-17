"""! @brief TensorFlow Fashion MNIST Example"""

############################
# Copyright: 2018 The TensorFlow Authors
# Gathered by: Sadegh Naderi
# Date created: 03.02.2024
# Path: ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense\Code\KeywordSpotting\errorHandler.py
# Version: 4.0
# Reviewed by: Sadegh Naderi
# Review Date: 04.02.2024
############################

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


##
# @mainpage TensorFlow Fashion MNIST Example
# @section intro_sec Introduction
# This script showcases the implementation of a neural network using TensorFlow for image classification on the Fashion MNIST dataset.
# The Fashion MNIST dataset consists of 28x28 grayscale images of 10 different fashion categories, making it a popular benchmark for image classification tasks.
#
# @section dependencies Dependencies
# The script relies on the following Python libraries:
# - TensorFlow (tf) - Deep learning framework for building and training neural networks.
# - NumPy (np) - Library for numerical operations and array manipulations.
# - Matplotlib (matplotlib) - Plotting and visualization library.
#
# @section usage Usage
# To run the script successfully, ensure that the required dependencies are installed in your Python environment. You can then execute the script
# to train and evaluate the neural network on the Fashion MNIST dataset.
#
# @section tensorflow_version TensorFlow Version
# - Version: 2.15.0
#
# @note Ensure your TensorFlow installation matches or is compatible with the specified version for optimal script execution.
#
# @section author Author
# - Created by Sadegh Naderi
#
# @section date Date
# - 08.02.2024



##
# @file clothingimgClassification.py
# @brief TensorFlow Fashion MNIST Example
#
# @section intro_sec Introduction
# This script demonstrates the use of TensorFlow to build, train, and evaluate a neural network for classifying images using the Fashion MNIST dataset.
#
# @section dependencies Dependencies
# - TensorFlow (tf)
# - NumPy (np)
# - Matplotlib (matplotlib)
#
# @section usage Usage
# To run the script, make sure to install the required dependencies and execute the script.
#
# @section description_of_file Description
# This Python file provides an example of training and evaluating a neural network using TensorFlow with the Fashion MNIST dataset. It covers data loading, model building, training, evaluation, and result visualization.
#
# @note This script is intended for educational purposes, showcasing the usage of TensorFlow, and may not be optimized for production use.
#
# @section tensorflow_version TensorFlow Version
# - Version: 2.15.0
#
# @section author Documentation Author
# - Created by Sadegh Naderi
#
# @section date Date
# - 09.02.2024

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

## @var fashionMnist
# @brief TensorFlow Fashion MNIST dataset.
#
# The `fashionMnist` variable represents the Fashion MNIST dataset, a collection of grayscale images of clothing items
# (28x28 pixels) used for training and testing machine learning models. It is loaded using the `tf.keras.datasets.fashion_mnist` module.
fashionMnist = tf.keras.datasets.fashion_mnist

## @var trainImages
# @brief Training images containing grayscale clothing images (28x28 pixels).
## @var trainLabels
# @brief Training labels representing the class indices for each image.
## @var testImages
# @brief Test images containing grayscale clothing images (28x28 pixels).
## @var testLabels
# @brief Test labels representing the class indices for each image.
(trainImages, trainLabels), (testImages, testLabels) = fashionMnist.load_data()

## @var classNames
# @brief Class names for Fashion MNIST dataset.
classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(trainImages.shape)
print(len(trainLabels))
print(trainLabels)
print(testImages.shape)
print(len(testLabels))

# Preprocess the data
plt.figure()
plt.imshow(trainImages[0])
plt.colorbar()
plt.grid(False)
plt.savefig('preprocessData.png')
plt.show()

# Normalize these values to fall within the range of 0 to 1
trainImages = trainImages / 255.0
testImages = testImages / 255.0

## @var figsize
# @brief Size of the figure for Matplotlib plots.
#
## @var cmap
# @brief Colormap used for visualizing images.
#
# The `cmap` variable specifies the color map for displaying images. In this context, it is used in Matplotlib's `imshow` function
# to define the color map for visualizing grayscale images.
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainImages[i], cmap=plt.cm.binary)
    plt.xlabel(classNames[trainLabels[i]])
plt.savefig('first25TrainingImages.png')
plt.show()

# Build the model

# Set up the layers

## @var model
# @brief Sequential model representing a neural network for image classification.
#
# The model consists of a Flatten layer, a Dense layer with ReLU activation,
# and a Dense layer without activation for class logits.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

## @var optimizer
# @brief Optimizer used during the compilation of the neural network model.
#
# The `optimizer` variable represents the optimization algorithm used to update the model parameters during training.
# Common optimizers include 'adam,' 'sgd' (Stochastic Gradient Descent), etc.
#
## @var loss
# @brief Loss function used during the compilation of the neural network model.
#
# The `loss` variable specifies the loss function that the model minimizes during training.
#
## @var metrics
# @brief List of metrics used to evaluate the performance of the neural network model during training.
#
# The `metrics` variable defines the evaluation metrics used to monitor the model's performance during training.
# Common metrics include 'accuracy' for classification tasks, 'mae' (Mean Absolute Error), etc.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model

# Feed the model
model.fit(trainImages, trainLabels, epochs=10)

# Evaluate accuracy

## @var testLoss
# @brief Test loss value obtained from evaluating the neural network model on the test set.
#
# The test loss represents the error between the model's predictions and the actual labels in the test set.
#
## @var testAcc
# @brief Test accuracy obtained from evaluating the neural network model on the test set.
#
# The test accuracy indicates the proportion of correctly classified instances in the test set.
testLoss, testAcc = model.evaluate(testImages, testLabels, verbose=2)

print('\nTest accuracy:', testAcc)

# Make predictions

## @var probabilityModel
# @brief Sequential model representing the neural network model with an additional Softmax layer.
#
# This model is used to obtain probability distributions over the classes for each input image.
#
## @var predictions
# @brief Predictions for the test set obtained using the probabilityModel.
#
# The predictions include the model's confidence scores for each class for each input image in the test set.
probabilityModel = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probabilityModel.predict(testImages)

print(predictions[0])
print(np.argmax(predictions[0]))
print(testLabels[0])


def plotImage(i, predictionsArray, trueLabel, img):
    """! Function to plot an image with predicted and true labels.
    This function takes an index `i` representing the position of the image in the dataset,
    an array of predictions (`predictionsArray`) providing the confidence scores for each class,
    the true label (`trueLabel`) for the image, and the image itself (`img`).

    The function plots the image using Matplotlib with certain visualization settings,
    including a binary color map. It then labels the image with the predicted and true class names,
    highlighting the predicted class in blue and any misclassification in red.

    @param i The index of the image in the dataset.
    @param predictionsArray Array of confidence scores for each class.
    @param trueLabel True label of the image.
    @param img The image data to be plotted.
    @return None
    """
    trueLabel, img = trueLabel[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predictedLabel = np.argmax(predictionsArray)
    if predictedLabel == trueLabel:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(classNames[predictedLabel],
                                          100 * np.max(predictionsArray),
                                          classNames[trueLabel]),
               color=color)


def plotValueArray(i, predictionsArray, trueLabel):
    """! Function to plot a bar chart representing the prediction array.

    This function takes an index `i` representing the position of the image in the dataset,
    an array of predictions (`predictionsArray`) providing the confidence scores for each class,
    and the true label (`trueLabel`) for the image.

    The function plots a bar chart using Matplotlib with certain visualization settings,
    including colors indicating the predicted and true classes.

    @param i The index of the image in the dataset.
    @param predictionsArray Array of confidence scores for each class.
    @param trueLabel True label of the image.
    @return None
    """

    trueLabel = trueLabel[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisPlot = plt.bar(range(10), predictionsArray, color="#777777")
    plt.ylim([0, 1])
    predictedLabel = np.argmax(predictionsArray)

    thisPlot[predictedLabel].set_color('red')
    thisPlot[trueLabel].set_color('blue')

## @var i
# @brief Index representing the position of the image in the dataset.
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plotImage(i, predictions[i], testLabels, testImages)
plt.subplot(1, 2, 2)
plotValueArray(i, predictions[i], testLabels)
plt.savefig(f'predictionPlot{i}.png')
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plotImage(i, predictions[i], testLabels, testImages)
plt.subplot(1, 2, 2)
plotValueArray(i, predictions[i], testLabels)
plt.savefig(f'predictionPlot{i}.png')
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.

## @var numRows
# @brief Number of rows in the grid for displaying test images.
#
## @var numCols
# @brief Number of columns in the grid for displaying test images.
#
## @var numImages
# @brief Total number of images to be displayed.
numRows = 5
numCols = 3
numImages = numRows * numCols
plt.figure(figsize=(2 * 2 * numCols, 2 * numRows))
for i in range(numImages):
    plt.subplot(numRows, 2 * numCols, 2 * i + 1)
    plotImage(i, predictions[i], testLabels, testImages)
    plt.subplot(numRows, 2 * numCols, 2 * i + 2)
    plotValueArray(i, predictions[i], testLabels)
plt.tight_layout()
plt.savefig('predictionPlots.png')
plt.show()

# Use the trained model
# Grab an image from the test dataset.

## @var img
# @brief Image data obtained from the test dataset.
img = testImages[1]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))
print(img.shape)

## @var predictionsSingle
# @brief Predictions for a single image obtained using the probabilityModel.
predictionsSingle = probabilityModel.predict(img)
print(predictionsSingle)

plotValueArray(1, predictionsSingle[0], testLabels)
_ = plt.xticks(range(10), classNames, rotation=45)
plt.savefig('singlePredictionPlot.png')
plt.show()

print(np.argmax(predictionsSingle[0]))

## @var savedModelPath
# @brief Path to save the model in the SavedModel format.
#
## @var converter
# @brief TensorFlow Lite Converter instance used to convert the SavedModel to TensorFlow Lite format.
#
## @var tfliteModel
# @brief TensorFlow Lite model obtained after converting the SavedModel.
#
## @var tfliteModelPath
# @brief Path to save the TensorFlow Lite model (.tflite) file.
#

# Save the model in the SavedModel format
savedModelPath = "./savedModel"
model.save(savedModelPath)

# Convert the SavedModel to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model(savedModelPath)
tfliteModel = converter.convert()

# Save the TFLite model to a file
tfliteModelPath = "./savedModel/model.tflite"
with open(tfliteModelPath, "wb") as f:
    f.write(tfliteModel)
