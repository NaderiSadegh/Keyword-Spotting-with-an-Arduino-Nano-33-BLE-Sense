"""! @brief Unit tests for the dataUtils module."""

############################
# Author: Sadegh Naderi
# Date created: 03.02.2024
# Path: ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense\Code\KeywordSpotting\testDataUtils.py
# Version: 4.0
# Reviewed by: Sadegh Naderi
# Review Date: 07.02.2024
############################


##
# ! @file testDataUtils.py
# @brief Unit tests for the dataUtils module.
#
# This file contains unit tests for the functions in the dataUtils module. The tests cover loading audio datasets,
# preprocessing audio datasets, and creating spectrogram datasets.
#
# @section authors Author
# - Created by Sadegh Naderi on 03.02.2024.
# - Modified by Sadegh Naderi on 04.02.2024.
#

import unittest
import tensorflow as tf
from dataUtils import loadDataset, preprocessAudioDataset, createSpectrogramDataset

class TestDataUtils(unittest.TestCase):
    """! TestDataUtils class for unit testing the dataUtils module.

    This class contains unit tests for the functions in the dataUtils module.
    """

    def testLoadDataset(self):
        """! Test loading dataset function.

        This test case checks if the loadDataset function successfully loads audio datasets
        and returns instances of TensorFlow datasets.
        """
        datasetPath = 'data/mini_speech_commands'
        trainDs, valDs = loadDataset(datasetPath, batchSize=64, validationSplit=0.2, seed=0, outputSequenceLength=16000)
        self.assertIsInstance(trainDs, tf.data.Dataset)
        self.assertIsInstance(valDs, tf.data.Dataset)

    def testPreprocessAudioDataset(self):
        """! Test preprocessing audio dataset function.

        This test case checks if the preprocessAudioDataset function successfully preprocesses
        audio datasets and returns the preprocessed dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices(([[1], [2]], [3, 4]))
        preprocessed_dataset = preprocessAudioDataset(dataset)

    def testCreateSpectrogramDataset(self):
        """! Test creating spectrogram dataset function.

        This test case checks if the createSpectrogramDataset function successfully creates
        spectrogram datasets from preprocessed audio datasets.
        """
        # Example dataset creation with preprocessed data
        audioData = [[1.0, 2.0], [3.0, 4.0]]
        labelData = [3, 4]

        # Preprocess the audio data to ensure it's a 1D tensor
        preprocessedAudioData = [tf.convert_to_tensor(waveform, dtype=tf.float32) for waveform in audioData]

        # Create the dataset with preprocessed data
        dataset = tf.data.Dataset.from_tensor_slices((preprocessedAudioData, labelData))

        # Apply the createSpectrogramDataset function
        spectrogram_dataset = createSpectrogramDataset(dataset)


if __name__ == '__main__':
    unittest.main()
