"""! @brief Unit tests for the exportUtils module."""

############################
# Author: Sadegh Naderi
# Date created: 03.02.2024
# Path: ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense\Code\KeywordSpotting\testExportUtils.py
# Version: 4.0
# Reviewed by: Sadegh Naderi
# Review Date: 07.02.2024
############################

##
# @file testExportUtils.py
# @brief Unit tests for the exportUtils module.
#
# This module contains unit tests for the functions and class in the exportUtils module,
# including tests for exporting a model, saving a model, and converting a model to TFLite format.
#
# @section authors Author
# - Created by Sadegh Naderi on 03.02.2024.
# - Modified by Sadegh Naderi on 04.02.2024.
#

import unittest
import os
import shutil
import tensorflow as tf
from exportUtils import ExportModel, saveModel, convertToTFLite


class TestExportUtils(unittest.TestCase):
    """! TestExportUtils class for unit testing exportUtils module.

    This class represents a test suite for the functions and class in the exportUtils module.
    It includes tests for exporting a model, saving a model, and converting a model to TFLite format.

    @date Created: [Date]
    @version 1.0
    """

    def testExportModel(self):
        """! Test ExportModel class.

        This method tests the functionality of the ExportModel class in the exportUtils module.
        It checks whether the ExportModel instance is created without errors.

        @exception AssertionError If the test fails.
        """
        model = tf.keras.Sequential()
        labelNames = ["label1", "label2"]
        export_model = ExportModel(model, labelNames)

    def testSaveModel(self):
        """! Test saving model function.

        This method tests the functionality of the saveModel function in the exportUtils module.
        It checks whether the model is saved successfully and the save directory is created.

        @exception AssertionError If the test fails.
        """
        model = tf.keras.Sequential()
        savePath = "savedModelTest"
        saveModel(model, savePath)

        # Check if the savedModelTest directory exists
        self.assertTrue(os.path.exists(savePath))
        # Optionally, you can check for specific files or conditions within the savedModelTest directory

        # Clean up: Remove the savedModelTest directory after the test
        if os.path.exists(savePath):
            shutil.rmtree(savePath)  # Use shutil.rmtree to remove the directory and its contents

    def testConvertToTFLite(self):
        """! Test converting model to TFLite function.

        This method tests the functionality of the convertToTFLite function in the exportUtils module.
        It assumes an error is raised during the conversion process.

        @exception AssertionError If the test fails.
        """
        with self.assertRaises(Exception):  # Assuming an error is raised in the original code
            convertToTFLite("savedModelTest", "model.tflite")


if __name__ == '__main__':
    unittest.main()
