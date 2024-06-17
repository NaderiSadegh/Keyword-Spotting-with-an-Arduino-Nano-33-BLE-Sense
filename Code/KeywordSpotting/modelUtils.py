"""! @brief Utility functions for building and compiling neural network models using TensorFlow."""

############################
# Author: Sadegh Naderi
# Date created: 01.02.2024
# Path: ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense\Code\KeywordSpotting\modelUtils.py
# Version: 3.0
# Reviewed by: Sadegh Naderi
# Review Date: 04.02.2024
############################


##
# @file modelUtils.py
# @brief Utility functions for building and compiling neural network models using TensorFlow.
#
# This module provides a function to build and compile neural network models with specified architectures
# using TensorFlow and Keras. It includes error handling mechanisms and leverages TensorFlow's capabilities for
# deep learning model construction.
#
# @section authors Author
# - Created by Sadegh Naderi on 01.02.2024.
# - Modified by Sadegh Naderi on 04.02.2024.
#


import tensorflow as tf
from tensorflow.keras import layers, models
import errorHandler


def buildModel(inputShape, numLabels, normLayer):
    """! Build and compile a neural network model.

    This function constructs a neural network model with a specific architecture and compiles it with
    appropriate optimization and loss functions.

    @param inputShape The shape of the input data.
    @param numLabels The number of output labels in the classification task.
    @param normLayer A normalization layer to be applied to the input data.
    @return A compiled Keras Sequential model.
    @exception errorHandler.errorBuildModel If an error occurs during the model construction.
    """
    try:
        model = models.Sequential([
            layers.Input(shape=inputShape),
            layers.Resizing(32, 32),
            normLayer,
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(numLabels),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        return model

    except Exception:
        errorHandler.errorBuildModel()
