"""! @brief Unit tests for the modelUtils module."""

############################
# Author: Sadegh Naderi
# Date created: 04.02.2024
# Path: ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense\Code\KeywordSpotting\testModelUtilst.py
# Version: 4.0
# Reviewed by: Sadegh Naderi
# Review Date: 07.02.2024
############################


##
# @file testModelUtilst.py
# @brief Unit tests for the modelUtils module.
#
# This module contains unit tests for the functions in the modelUtils module,
# specifically testing the buildModel function.
#
# @section authors Author
# - Created by Sadegh Naderi on 04.02.2024.
# - Modified by Sadegh Naderi on 04.02.2024.
#

import unittest
import tensorflow as tf
from tensorflow.keras import layers
from modelUtils import buildModel


class TestModelUtils(unittest.TestCase):
    """! TestModelUtils class for unit testing modelUtils module.

    This class represents a test suite for the functions in the modelUtils module.
    It includes tests for building a model using the buildModel function.
    """

    def testBuildModel(self):
        """! Test building model function.

        This method tests the functionality of the buildModel function in the modelUtils module.
        It checks whether the model is built successfully and if the layers match the expected structure.

        @exception AssertionError If the test fails.
        """
        inputShape = (32, 32, 1)
        numLabels = 8  # Adjust the number of labels based on your actual model
        normLayer = tf.keras.layers.Normalization(axis=-1)
        model = buildModel(inputShape, numLabels, normLayer)

        self.assertIsInstance(model, tf.keras.models.Sequential)

        # Check each layer in the model
        expectedLayers = [
            layers.Resizing(32, 32),
            normLayer,
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(numLabels)
        ]

        for i, (expected_layer, actual_layer) in enumerate(zip(expectedLayers, model.layers)):
            with self.subTest(f"Testing layer {i}"):
                print(f"Actual Layer {i}: {type(actual_layer).__name__}")
                self.assertEqual(type(expected_layer).__name__, type(actual_layer).__name__)

                if hasattr(expected_layer, 'input_shape'):
                    self.assertEqual(expected_layer.input_shape, actual_layer.input_shape)
                if hasattr(expected_layer, 'units'):
                    self.assertEqual(expected_layer.units, actual_layer.units)
                if hasattr(expected_layer, 'activation'):
                    self.assertEqual(expected_layer.activation.__name__, actual_layer.activation.__name__)


if __name__ == '__main__':
    unittest.main()
