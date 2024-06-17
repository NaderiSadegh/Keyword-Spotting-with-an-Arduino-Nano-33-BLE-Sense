"""! @brief Speech recognition workflow using TensorFlow for audio dataset."""

############################
# Author: Sadegh Naderi
# Date created: 01.02.2024
# Path: ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense\Code\KeywordSpotting\KeywordSpotting.py
# Version: 3.0
# Reviewed by: Sadegh Naderi
# Review Date: 04.02.2024
############################


##
# @mainpage Keyword Spotting with Arduino Nano 33 BLE Sense
#
# @section author Author
# - Created by Sadegh Naderi on 04.02.2024.
# - Modified by Sadegh Naderi on 05.02.2024.
#
# @section intro_sec Introduction
#
# This documentation outlines the Keyword Spotting project using TensorFlow on Arduino Nano 33 BLE Sense.
# The project involves recognizing specific keywords in audio data, and it covers the entire workflow from
# dataset loading to model evaluation and deployment on Arduino Nano 33 BLE Sense.
#
# @section documentation Workflow Overview
#
# The script in `KeywordSpotting.py` demonstrates the speech recognition workflow using TensorFlow. It includes:
# 1. Loading and preprocessing an audio dataset.
# 2. Creating spectrogram datasets from the preprocessed audio data.
# 3. Building and training a neural network model.
# 4. Evaluating the model's performance.
# 5. Exporting the model for inference, saving it, and converting it to TFLite format.
#
# @section dataset Dataset Information
#
# The audio dataset is expected to be organized in a specific structure. The dataset path and format are crucial
# for successful loading and preprocessing. Ensure that the dataset includes the necessary keywords for spotting.
#
# @section model Model Architecture
#
# The model architecture is constructed using TensorFlow, with a focus on efficiency for deployment on Arduino Nano 33 BLE Sense.
# It utilizes spectrogram data for training and inference. The documentation provides insights into the model's structure
# and key considerations for optimization.
#
# The time-domain signals of the waveforms undergo a transformation into
# time-frequency-domain signals through the computation of the short-time
# Fourier transform (STFT)
#
# @image html audioWaveSpectrogramYes.png "Audio Waveforms and Spectrogram" width=500px
#
# the model evaluation is done based on the accuracy and loss of the training and validation data.
#
# @image html trainingProgress.png "Training Progress" width=500px
#
# @section module Modules
#
# The project has the following Python files:
#
# The following links provide access to documentation files related to the project.
#
# - 'KeywordSpotting.py': Main script showcasing the speech recognition workflow.
# - 'dataUtils.py': Utility functions for dataset loading and preprocessing.
# - 'exportUtils.py': Utility functions for model exporting and saving.
# - 'modelUtils.py': Utility functions for building and training the model.
# - 'testDataUtils.py': Unit tests for dataUtils module.
# - 'testExportUtils.py': Unit tests for exportUtils module.
# - 'testModelUtilst.py': Unit tests for modelUtils module.
# - 'errorHandler.py': Error handling functions for various modules.
#
# @section start Getting Started
#
# @warning Before starting, ensure that you have the required packages and dependencies installed. Detailed information is provided
# in the documentation of each file. Additionally, the dataset structure and model configurations should align with project requirements.
# Once set up, follow the step-by-step instructions in the codebase.
#
# @note For specific details on dataset structure, model architecture, and Arduino deployment, refer to individual sections
# and comments within the code.


##
# @file speechRecognition.py
# @brief Speech recognition workflow using TensorFlow for audio dataset.
#
# This script demonstrates a speech recognition workflow using TensorFlow. It includes the following steps:
# 1. Load and preprocess an audio dataset.
# 2. Create spectrogram datasets from the preprocessed audio data.
# 3. Build and train a neural network model.
# 4. Evaluate the model's performance.
# 5. Export the model for inference, save it, and convert it to TFLite format.
#
# @section authors Author
# - Created by Sadegh Naderi on 01.02.2024.
# - Modified by Sadegh Naderi on 04.02.2024.
#


import numpy as np
import tensorflow as tf
import pathlib
from dataUtils import loadDataset, preprocessAudioDataset, createSpectrogramDataset
from modelUtils import buildModel
from exportUtils import ExportModel, saveModel, convertToTFLite
import errorHandler


## @var try
#   @brief Exception handling block for the main execution.
#
#   This 'try' block encompasses the main execution of the speechRecognition.py script, containing the entire
#   workflow of loading, preprocessing, building, training, evaluating, exporting, saving, and converting the model.
#   Exception handling mechanisms are employed to catch and handle errors that may occur during each step.
try:
    # Set the seed value for experiment reproducibility.
    ## @var seed
    #   @brief Seed value for random number generation.
    #
    #   This seed is set to ensure reproducibility of experiments involving random processes.
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Import the dataset
    ## @var datasetPath
    #   @brief Path to the audio dataset directory.
    #
    #   The variable holds the path to the directory containing the audio dataset for speech recognition.
    datasetPath = 'data/mini_speech_commands'

    # Load and preprocess the dataset
    try:
        ## @var batchSize
        #   @brief Batch size for loading and processing datasets.
        #
        #   This variable specifies the size of batches used during dataset loading and processing.

        ## @var validationSplit
        #   @brief Fraction of the dataset allocated to the validation set.
        #
        #   This variable specifies the fraction of the dataset used for validation during training.

        ## @var outputSequenceLength
        #   @brief Output sequence length for audio data.
        #
        #   This variable specifies the desired output sequence length for audio data.

        ## @var trainDs
        #   @brief TensorFlow Dataset containing the training audio dataset.
        #
        #   This variable holds the TensorFlow Dataset containing the training audio dataset for model training.
        #
        ## @var valDs
        #   @brief TensorFlow Dataset containing the validation audio dataset.
        #
        #   This variable holds the TensorFlow Dataset containing the validation audio dataset, which is used for
        #   evaluating the trained model's performance on unseen data during the validation phase.

        ## @var labelNms
        #   @brief List of class names extracted from the training dataset.
        #
        #   This variable holds the list of class names extracted from the training dataset.

        ## @var labelNames
        #   @brief Numpy array of class names.
        #
        #   This variable holds a numpy array containing class names, converted from the list of class names.

        ## @var valDs
        #   @brief TensorFlow Dataset containing the validation audio dataset.
        #
        #   This dataset is loaded and preprocessed for validating the trained speech recognition model.

        trainDs, valDs = loadDataset(datasetPath, batchSize=64, validationSplit=0.2, seed=0, outputSequenceLength=16000)
        labelNms = trainDs.class_names
        labelNames = np.array(labelNms)
        trainDs = preprocessAudioDataset(trainDs)

        valDs = preprocessAudioDataset(valDs)

    except Exception:
        errorHandler.errorLoadDataset()

    # Create spectrogram datasets
    try:
        ## @var trainSpectrogramDs
        #   @brief TensorFlow Dataset containing spectrogram data for training.
        #
        #   This dataset is created by converting the training audio dataset into spectrograms for model training.
        trainSpectrogramDs = createSpectrogramDataset(trainDs)

        ## @var valSpectrogramDs
        #   @brief TensorFlow Dataset containing spectrogram data for validation.
        #
        #   This dataset is created by converting the validation audio dataset into spectrograms for model validation.
        valSpectrogramDs = createSpectrogramDataset(valDs)

        ## @var exampleSpectrograms
        #   @brief Example spectrograms from the training set.
        #
        #   An example set of spectrograms is extracted for visualization or testing purposes.
        for exampleSpectrograms, exampleSpectLabels in trainSpectrogramDs.take(1):
            break

    except Exception:
        errorHandler.errorCreateSpectrogram()

    # Build and train the model
    try:
        ## @var inputShape
        #   @brief Shape of input spectrograms for the model.
        #
        #   This variable holds the shape of input spectrograms, extracted from the example spectrograms.
        inputShape = exampleSpectrograms.shape[1:]

        ## @var numLabels
        #   @brief Number of labels in the dataset.
        #
        #   This variable holds the total number of labels based on the extracted class names.
        numLabels = len(labelNames)

        ## @var normLayer
        #   @brief Normalization layer for the model.
        #
        #   This variable holds an instance of the Normalization layer for normalizing input spectrograms.
        normLayer = tf.keras.layers.Normalization()

        ## @var model
        #   @brief TensorFlow Sequential model for speech recognition.
        #
        #   This variable holds the instance of the Sequential model used for training the speech recognition model.
        model = buildModel(inputShape, numLabels, normLayer)

        ## @var history
        #   @brief Training history of the model.
        #
        #   The variable stores the training history of the model, including loss and accuracy over epochs.
        model.summary()

        ## @var optimizer
        #   @brief Model optimizer for training.
        #
        #   This variable holds the instance of the Adam optimizer used during model training.
        optimizer = tf.keras.optimizers.Adam()

        ## @var loss
        #   @brief Loss function for model training.
        #
        #   This variable holds the instance of the SparseCategoricalCrossentropy loss function.
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        ## @var metrics
        #   @brief Evaluation metrics for the model.
        #
        #   This variable holds the list of evaluation metrics, including accuracy, used during model training.
        metrics = ['accuracy']

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        ## @var EPOCHS
        #   @brief Number of training epochs.
        #
        #   This variable specifies the number of training epochs during model training.
        EPOCHS = 10
        history = model.fit(
            trainSpectrogramDs.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE),
            validation_data=valSpectrogramDs.cache().prefetch(tf.data.AUTOTUNE),
            epochs=EPOCHS,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        )

    except Exception:
        errorHandler.errorBuildModel()

    # Evaluate the model performance
    try:
        ## @var testDs
        #   @brief Subset of the validation spectrogram dataset for testing.
        #
        #   This variable holds a subset of the validation spectrogram dataset for testing model performance.
        testDs = valSpectrogramDs.shard(num_shards=2, index=0)
        model.evaluate(testDs.cache().prefetch(tf.data.AUTOTUNE), return_dict=True)

    except Exception:
        errorHandler.errorEvaluateModel()

    # Export the model with preprocessing
    try:
        ## @var export
        #   @brief ExportModel instance for saving the model.
        #
        #   This variable holds an instance of the ExportModel class for exporting the trained model.
        export = ExportModel(model, labelNms)
        print(export(tf.constant(str(pathlib.Path(datasetPath) / 'no/01bb6a2a_nohash_0.wav'))))

    except Exception:
        errorHandler.errorExportModel()

    # Save the model
    try:
        ## @var savedModelPath
        #   @brief Path to save the trained model.
        #
        #   This variable holds the path where the trained model will be saved in the SavedModel format.
        savedModelPath = "savedModel"
        saveModel(model, savedModelPath)

    except Exception:
        errorHandler.errorSaveModel()

    # Convert the SavedModel to TFLite
    try:
        ## @var tfliteModelPath
        #   @brief Path to save the TFLite model.
        #
        #   This variable holds the path where the TFLite model will be saved after conversion.
        tfliteModelPath = f"{savedModelPath}/model.tflite"
        convertToTFLite(savedModelPath, tfliteModelPath)

    except Exception:
        errorHandler.errorTFLiteConversion()

except Exception:
    errorHandler.handleError()

