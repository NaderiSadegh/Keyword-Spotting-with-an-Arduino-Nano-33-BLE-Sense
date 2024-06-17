"""! @brief Utility functions for loading and preprocessing audio datasets using TensorFlow."""

############################
# Author: Sadegh Naderi
# Date created: 01.02.2024
# Path: ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense\Code\KeywordSpotting\dataUtils.py
# Version: 3.0
# Reviewed by: Sadegh Naderi
# Review Date: 04.02.2024
############################


##
# @file dataUtils.py
#
# @brief Utility functions for loading and preprocessing audio datasets using TensorFlow.
#
# This module provides functions to load audio datasets, preprocess audio data, and create spectrogram datasets
# using TensorFlow. It includes error handling mechanisms and leverages TensorFlow's capabilities for audio data handling.
#
# @section authors Author
# - Created by Sadegh Naderi on 01.02.2024.
# - Modified by Sadegh Naderi on 04.02.2024.
#


import tensorflow as tf
import pathlib
import numpy as np
import errorHandler


def loadDataset(datasetPath, batchSize=64, validationSplit=0.2, seed=0, outputSequenceLength=16000):
    """! Load audio dataset from the specified path.

    This function loads an audio dataset from the given path. If the dataset is not present, it is downloaded
    from a specified URL. The dataset is then split into training and validation sets.

    @param datasetPath The path to the audio dataset.
    @param batchSize The batch size for the datasets (default: 64).
    @param validationSplit The fraction of the dataset to allocate to the validation set (default: 0.2).
    @param seed The seed value for random operations (default: 0).
    @param outputSequenceLength The output sequence length for the audio data (default: 16000).
    @return A tuple containing training and validation datasets.
    @exception errorHandler.errorLoadDataset If an error occurs while loading the dataset.
    """
    try:
        dataDir = pathlib.Path(datasetPath)

        if not dataDir.exists():
            tf.keras.utils.get_file(
                'mini_speech_commands.zip',
                origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                extract=True,
                cache_dir='.',
                cache_subdir='data'
            )

        else:
            print("The dataset already exists")

        commands = np.array(tf.io.gfile.listdir(str(dataDir)))
        commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
        print('Commands:', commands)

        trainDs, valDs = tf.keras.utils.audio_dataset_from_directory(
            directory=dataDir,
            batch_size=batchSize,
            validation_split=validationSplit,
            seed=seed,
            output_sequence_length=outputSequenceLength,
            subset='both'
        )

        return trainDs, valDs

    except Exception:
        errorHandler.errorLoadDataset()


def preprocessAudioDataset(dataset):
    """! Preprocess audio dataset by squeezing dimensions.

    This function squeezes the last dimension of audio data in the dataset.

    @param dataset The input audio dataset.
    @return The preprocessed audio dataset.
    @exception errorHandler.errorProcessAudio If an error occurs during audio data preprocessing.
    """
    def squeeze(audio, labels):
        try:
            audio = tf.squeeze(audio, axis=-1)
            return audio, labels
        except Exception:
            errorHandler.errorProcessAudio()

    dataset = dataset.map(squeeze, tf.data.AUTOTUNE)
    return dataset


def createSpectrogramDataset(dataset):
    """! Create spectrogram dataset from the input audio dataset.

    This function computes the spectrogram of audio waveforms in the dataset using TensorFlow's signal processing
    functions.

    @param dataset The input audio dataset.
    @return The spectrogram dataset.
    @exception errorHandler.errorSpectrogram If an error occurs during spectrogram creation.
    """
    def getSpectrogram(waveform):
        try:
            spectrogram = tf.signal.stft(
                waveform, frame_length=255, frame_step=128)
            spectrogram = tf.abs(spectrogram)
            spectrogram = spectrogram[..., tf.newaxis]
            return spectrogram
        except Exception:
            errorHandler.errorSpectrogram()

    dataset = dataset.map(
        map_func=lambda audio, label: (getSpectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset
