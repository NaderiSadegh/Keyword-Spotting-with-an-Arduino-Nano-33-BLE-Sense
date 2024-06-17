"""! @brief Utility functions for exporting, saving, and converting TensorFlow models."""

############################
# Author: Sadegh Naderi
# Date created: 01.02.2024
# Path: ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense\Code\KeywordSpotting\exportUtils.py
# Version: 3.0
# Reviewed by: Sadegh Naderi
# Review Date: 04.02.2024
############################


##
# @file exportUtils.py
# @brief Utility functions for exporting, saving, and converting TensorFlow models.
#
# This module provides functions to export a TensorFlow model, save a model to a specified path, and convert a
# SavedModel to TensorFlow Lite format. It also includes a class `ExportModel` for exporting models with
# preprocessing.
#
#
# @section authors Author
# - Created by Sadegh Naderi on 01.02.2024.
# - Modified by Sadegh Naderi on 04.02.2024.
#


import tensorflow as tf
import errorHandler


class ExportModel(tf.Module):
    """! ExportModel class for exporting TensorFlow models with preprocessing.

    This class represents a TensorFlow module for exporting models with preprocessing steps.
    It includes methods to handle export initialization, calling the model for predictions,
    and computing the spectrogram of audio waveforms.
    """
    def __init__(self, model, labelNames):
        """! Constructor for ExportModel.

        Initializes an instance of the ExportModel class.

        @param model The TensorFlow model to be exported.
        @param labelNames The list of label names used in classification.
        @exception errorHandler.errorExportInit If an error occurs during export initialization.
        """
        self.model = model
        self.labelNames = labelNames

        try:
            self.__call__.get_concrete_function(
                x=tf.TensorSpec(shape=(), dtype=tf.string))
            self.__call__.get_concrete_function(
                x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))
        except Exception:
            errorHandler.errorExportInit()

    @tf.function
    def __call__(self, x):
        """! Call method for making predictions with the model.

        This method takes an input tensor `x` and performs necessary preprocessing
        before passing it through the model for predictions.

        @param x The input tensor containing audio data.
        @return A dictionary containing predictions, class IDs, and class names.
        @exception errorHandler.errorExportCall If an error occurs during prediction.
        """
        try:
            if x.dtype == tf.string:
                x = tf.io.read_file(x)
                x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
                x = tf.squeeze(x, axis=-1)
                x = x[tf.newaxis, :]

            x = self.getSpectrogram(x)  # Call getSpectrogram from the class instance
            result = self.model(x, training=False)

            classIds = tf.argmax(result, axis=-1)
            classNames = tf.gather(self.labelNames, classIds)
            return {'predictions': result,
                    'classIds': classIds,
                    'classNames': classNames
                    }

        except Exception:
            errorHandler.errorExportCall()

    def getSpectrogram(self, waveform):
        """! Compute spectrogram of audio waveform.

        This method computes the spectrogram of an audio waveform using TensorFlow's signal processing functions.

        @param waveform The input audio waveform tensor.
        @return The computed spectrogram tensor.
        @exception errorHandler.errorExportSpectrogram If an error occurs during spectrogram computation.
        """
        try:
            spectrogram = tf.signal.stft(
                waveform, frame_length=255, frame_step=128)
            spectrogram = tf.abs(spectrogram)
            spectrogram = spectrogram[..., tf.newaxis]
            return spectrogram
        except Exception:
            errorHandler.errorExportSpectrogram()


def saveModel(model, exportPath):
    """! Save a TensorFlow model to a specified path.

    This function saves a TensorFlow model to a specified path using the SavedModel format.

    @param model The TensorFlow model to be saved.
    @param exportPath The path where the model should be saved.
    @exception errorHandler.errorSaveModel If an error occurs during model saving.
    """
    try:
        tf.saved_model.save(model, exportPath)
    except Exception:
        errorHandler.errorSaveModel()


def convertToTFLite(savedModelPath, tfliteModelPath):
    """! Convert a SavedModel to TensorFlow Lite format.

    This function converts a SavedModel to TensorFlow Lite format and saves the resulting
    TFLite model to a specified path.

    @param savedModelPath The path to the SavedModel.
    @param tfliteModelPath The path where the TFLite model should be saved.
    @exception errorHandler.errorTFLiteConversion If an error occurs during TFLite conversion.
    """
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(savedModelPath)
        tfliteModel = converter.convert()

        with open(tfliteModelPath, "wb") as f:
            f.write(tfliteModel)
    except Exception as e:
        errorHandler.errorTFLiteConversion()
