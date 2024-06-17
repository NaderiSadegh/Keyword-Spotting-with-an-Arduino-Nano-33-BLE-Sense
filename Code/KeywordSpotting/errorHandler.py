"""! @brief Error handling functions for various modules."""

############################
# Author: Sadegh Naderi
# Date created: 03.02.2024
# Path: ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense\Code\KeywordSpotting\errorHandler.py
# Version: 3.0
# Reviewed by: Sadegh Naderi
# Review Date: 04.02.2024
############################


##
# @file errorHandler.py
# @brief Error handling functions for various modules.
#
# This module provides error handling functions for different modules in the project.
# Each function raises a RuntimeError with a specific error message related to its context.
#
# @section authors Author
# - Created by Sadegh Naderi on 03.02.2024.
# - Modified by Sadegh Naderi on 04.02.2024.
#
#

import logging

# Configure the logging module
logging.basicConfig(filename='errorHandler.log', level=logging.ERROR)

def logandRaiseError(message):
    """!Logs an error message using the logging module and raises a RuntimeError.
    This function logs the provided error message using the logging module with
    the severity level set to ERROR. It then raises a RuntimeError with the
    same error message.

    @param message The error message to log and raise.
    type: message str

    @see https://docs.python.org/3/library/logging.html
    """
    logging.error(message)
    raise RuntimeError(message)

def errorLoadDataset():
    """! Raise an error for dataset loading failure.

    This function raises a RuntimeError when there is a failure to download and load the dataset.
    It suggests checking the internet connection and dataset path.

    @exception RuntimeError Always raised with a descriptive error message.
    """
    logandRaiseError("Failed to download and load the dataset. Check your internet connection and dataset path.")


def errorProcessAudio():
    """! Raise an error for audio data processing failure.

    This function raises a RuntimeError when there is a failure to process audio data.
    It advises ensuring the audio files are in the correct format.

    @exception RuntimeError Always raised with a descriptive error message.
    """
    logandRaiseError("Failed to process audio data. Ensure the audio files are in the correct format.")


def errorSpectrogram():
    """! Raise an error for spectrogram creation failure.

    This function raises a RuntimeError when there is a failure to create a spectrogram.
    It suggests checking the audio data for anomalies.

    @exception RuntimeError Always raised with a descriptive error message.
    """
    logandRaiseError("Failed to create spectrogram. Check the audio data for anomalies.")


def errorExportInit():
    """! Raise an error during ExportModel initialization.

    This function raises a RuntimeError when there is an error during the initialization of the ExportModel class.
    It advises ensuring the model and label information are valid.

    @exception RuntimeError Always raised with a descriptive error message.
    """
    logandRaiseError("Error during ExportModel initialization. Ensure the model and label information are valid.")


def errorExportCall():
    """! Raise an error during ExportModel class invocation.

    This function raises a RuntimeError when there is an error during the invocation of the ExportModel class.
    It suggests verifying the input data and model compatibility.

    @exception RuntimeError Always raised with a descriptive error message.
    """
    logandRaiseError("Error during calling the ExportModel class. Verify the input data and model compatibility.")


def errorExportSpectrogram():
    """! Raise an error during exporting spectrogram.

    This function raises a RuntimeError when there is an error during exporting a spectrogram.
    It suggests checking the audio data for abnormalities.

    @exception RuntimeError Always raised with a descriptive error message.
    """
    logandRaiseError("Error during exporting spectrogram. Check the audio data for abnormalities.")


def errorTFLiteConversion():
    """! Raise an error during TFLite conversion.

    This function raises a RuntimeError when there is an error during TFLite conversion.
    It suggests reviewing the model architecture and compatibility.

    @exception RuntimeError Always raised with a descriptive error message.
    """
    logandRaiseError("Error during TFLite conversion. Review the model architecture and compatibility.")


def errorBuildModel():
    """! Raise an error for model building failure.

    This function raises a RuntimeError when there is a failure to build the model.
    It suggests reviewing the model architecture and layer configurations.

    @exception RuntimeError Always raised with a descriptive error message.
    """
    logandRaiseError("Failed to build the model. Review the model architecture and layer configurations.")


def errorSaveModel():
    """! Raise an error during model saving.

    This function raises a RuntimeError when there is an error during model saving.
    It suggests ensuring the export path is valid and has write permissions.

    @exception RuntimeError Always raised with a descriptive error message.
    """
    logandRaiseError("Error during model saving. Ensure the export path is valid and has write permissions.")


def errorCreateSpectrogram():
    """! Raise an error for spectrogram creation failure during preprocessing.

    This function raises a RuntimeError when there is a failure to create a spectrogram during preprocessing.
    It suggests checking the audio data for anomalies.

    @exception RuntimeError Always raised with a descriptive error message.
    """
    logandRaiseError("Failed to create spectrogram. Check the audio data for anomalies during preprocessing.")


def errorEvaluateModel():
    """! Raise an error during model evaluation.

    This function raises a RuntimeError when there is an error during model evaluation.
    It suggests checking the dataset and model compatibility.

    @exception RuntimeError Always raised with a descriptive error message.
    """
    logandRaiseError("Error during model evaluation. Check the dataset and model compatibility.")


def errorExportModel():
    """! Raise an error during model export.

    This function raises a RuntimeError when there is an error during model export.
    It suggests verifying the input data and label information for correctness.

    @exception RuntimeError Always raised with a descriptive error message.
    """
    logandRaiseError("Error during model export. Verify the input data and label information for correctness.")


def handleError():
    """! Raise a generic unexpected error.

    This function raises a RuntimeError for any unexpected error.
    It advises reviewing the error messages for more details.

    @exception RuntimeError Always raised with a generic descriptive error message.
    """
    logandRaiseError("An unexpected error occurred. Please review the error messages.")
