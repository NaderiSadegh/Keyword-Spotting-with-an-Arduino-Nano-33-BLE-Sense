# GitHubTemplate

Name of the project

- Keyword Spotting with an Arduino Nano 33 BLE Sense

  Supervisor: Prof. Dr. Elmar Wings

Contributors:

- Shakywar, Achal, 7025278
- Naderi, Sadegh, 7024414
- Ghansletwala, Malik Al Ashter, 7025306

# Short Description of the Project

This project explores the application of Tiny Machine Learning (TinyML) to create a keyword spotting system using an Arduino Nano 33 BLE Sense. The goal of this project is to train the Arduino Nano 33 BLE Sense to recognize specific keywords from a dataset and trigger an RGB LED to blink whenever it senses one of the trained keywords from a sample of the trained data.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Directory Structure](#directory-structure)
- [Training Process](#training-process)
- [Analysis and Visualization](#analysis-and-visualization)
- [Deployment](#deployment)
- [Usage](#usage)
- [Results](#results)

## Prerequisites

Before you get started with this project, make sure you have the following prerequisites in place:

- Arduino Nano 33 BLE Sense board
- Arduino IDE with the ArduinoBLE library installed
- Trained dataset of keywords
- Knowledge of TinyML and TensorFlow

## Getting Started

To set up this project, follow these steps:

1. Clone or download this repository.
2. Connect your Arduino Nano 33 BLE Sense to your computer.
3. Open the Arduino IDE and load the provided sketch.
4. Upload the sketch to your Arduino Nano 33 BLE Sense.
5. Follow the deployment and usage instructions to train and run the keyword spotting system.

## Directory Structure

The project repository is organized as follows:

* [report](./report): Report directory

  * [KeywordSpotting.pdf](./report/KeywordSpotting.pdf): The main report pdf
  * [KeywordSpotting.tex](./report/KeywordSpotting.tex) 

  * [Contents](./report/Contents): tex files for contents of each chapter
    * [en](./reports/Contents/en)
      * [BillofMaterials.tex](./report/Contents/en/BillofMaterials.tex): Hardware and Sofware BoM, requirements
      * [Conclusion.tex](./report/Contents/en/Conclusion.tex)
      * [DataDescription.tex](./report/Contents/en/DataDescription.tex): Description of the dataset
      * [DataMining.tex](./report/Contents/en/DataMining.tex): Machine Learning Algorithm (CNN)
      * [DataTransformandMining.tex](./report/Contents/en/DataTransformandMining.tex): Data transformation and Data Mining in KDD
      * [Deployment.tex](./report/Contents/en/Deployment.tex): Deployment in KDD
      * [DevelopmentEnvironment.tex](./report/Contents/en/DevelopmentEnvironment.tex): Development environment, how to create manual
      * [DocumentationDevelopment.tex](./report/Contents/en/DocumentationDevelopment.tex), Documentation developer for tech development
      * [Domain.tex](./report/Contents/en/Domain.tex): Domain knowledge for data and problem
      * [HardwareDescription.tex](./report/Contents/en/HardwareDescription.tex): Description of the board Arduino Nano 33 BLE Sense
      * [Introduction.tex](./report/Contents/en/Introduction.tex)
      * [KDDIntroduction.tex](./report/Contents/en/KDDIntroduction.tex) Introduction of the KDD process
      * [Numpy.tex](./report/Contents/en/Numpy.tex) Numpy package explanation
      * [ProgramFlowchart.tex](./report/Contents/en/ProgramFlowchart.tex): Program flowchart and explanation
      * [Results.tex](./report/Contents/en/Results.tex): The achieved results and analysis
      * [SoftwareDescription.tex](./report/Contents/en/SoftwareDescription.tex): Description of the software
      * [TensorFlow.tex](./report/Contents/en/TensorFlow.tex): TensorFlow package explanation
      * [testSoftware.tex](./report/Contents/en/testSoftware.tex): Explanation of the test modules for software, simple manual for testing and automation of the testing process

  * [Code](./report/Code/): Directory for example code files used in the report

    * [CNN](./report/Code/CNN): Example for Data Mining
      * [CNNDataMining.py](./report/Code/CNN/CNNDataMining.py)
      * [Documentation](./report/Code/CNN/Documentation/)
        * [Doxyfile](./report/Code/CNN/Documentation/Doxyfile): Configuration file for Doxygen
        * [html](./report/Code/CNN/Documentation/html/): Doxygen Documentation
          * [index.html](./report/Code/CNN/Documentation/html/index.html)
    * [Deployment](./report/Code/Deployment)
      * [DeployableCode.py](./report/Code/Deployment/DeployableCode.py)
      * [TensorflowLiteSampleCode.py](./report/Code/Deployment/TensorflowLiteSampleCode.py)
    * [DevelopmentEnvExample](./report/Code/DevelopmentEnvExample): Development environment example
      * [HelloWorld.py](./report/Code/DevelopmentEnvExample/HelloWorld.py)
      * [HelloWorldDoxy](./report/Code/DevelopmentEnvExample/HelloWorldDoxy/):  Doxygen documentation
        * [index.html](./report/Code/DevelopmentEnvExample/HelloWorldDoxy/html/index.html) 
    * [Numpy](./report/Code/Numpy)
      * [NumPy.py](./report/Code/Numpy/NumPy.py): Numpy package exmple
    * [TensorFlow](./report/Code/TensorFlow): TensorFlow package example
      * [clothingimgClassification.py](./report/Code/TensorFlow/clothingimgClassification.py): Classify clothing items
      * [Doxygen](./report/Code/TensorFlow/Doxygen): Doxygen documentation
        * [index.html](./report/Code/TensorFlow/Doxygen/html/index.html)
    * [testExample](./report/Code/testExample): Simple example for software tests
      * [calc.py](./report/Code/testExample/calc.py): Addition Calculator
      * [testCalc.py](./report/Code/testExample/testCalc.py): Test of addition Calculator
      * [Doxygen](./report/Code/testExample/Doxygen): Doxygen documentation
        * [index.html](./report/Code/testExample/Doxygen/html/index.html)
    * [testhardware](./report/Code/testhardware): Hardware tests
      * [microphone.c](./report/Code/testhardware/microphone.c)

  * [images](./report/Images/): Directory for images used in the report

 * Documents

   * [MyLiterarture.bib](./Documents/MyLiterature.bib): The .bib file used both for report and literature review

  * Code (the code used for the project)
    * [KeywordSpotting](./Code/KeywordSpotting): Python code for the project directory
      * [KeywordSpotting.py](./Code/KeywordSpotting/KeywordSpotting.py): Main Python execution code
      * [dataUtils.py](./Code/KeywordSpotting/dataUtils.py): Data downloading and processing module
      * [modelUtils.py](./Code/KeywordSpotting/modelUtils.py): Model builder module
      * [exportUtils.py](./Code/KeywordSpotting/exportUtils.py): Exporting module: saving model etc.
      * [testDataUtils.py](./Code/KeywordSpotting/testDataUtils.py): Test for dataUtils module
      * [testModelUtilst.py](./Code/KeywordSpotting/testModelUtilst.py): Test for modelUtils module
      * [testExportUtils.py](./Code/KeywordSpotting/testExportUtils.py): Test for exportUtils module
      * [Doxygen](./Code/KeywordSpotting/Doxygen): Doxygen documentation of the python files
        * [index.html](./Code/KeywordSpotting/Doxygen/html/index.html)
      * [savedModel](./Code/KeywordSpotting/savedModel): Directory of the saved model
        * [model.tflite](./Code/KeywordSpotting/savedModel/model.tflite) The TensorFlow Lite model
        * [tinyConv.cc](./Code/KeywordSpotting/savedModel/tinyConv.cc) The converted C header model
      * [envRequirements](./Code/KeywordSpotting/envRequirements): Environment requirements
        * [environment.yml](./Code/KeywordSpotting/envRequirements/environment.yml): Automating the environment creation process with yml file
        * [requirements.txt](./Code/KeywordSpotting/envRequirements/requirements.txt): Automating the environment creation process with txt file
  
* [Poster](./Poster): The project poster
  * [KeywordSpottingPoster.pdf](./Poster/KeywordSpottingPoster.pdf)
  * [KeywordSpottingPoster.tex](./Poster/KeywordSpottingPoster.tex)




## Data transformation

<div class="image">
  <img class=center src="https://github.com/Wings-hub/ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense/blob/main/report/Images/Results/audioWaveSpectrogramYes.png" alt="" width="500px">
  <div class="caption">Figure 1: Audio waveform of the keyword "yes" is transformed to a spectrogram</div>
</div>

Converting audio to a spectrogram involves analyzing one-second audio snippets through a loop. Each 30-millisecond segment undergoes fast Fourier transform (FFT) with a 20-millisecond overlap, generating a 2D array that represents the entire audio sample. This array, referred to as a spectrogram, captures the intensity of various frequency components over time. Subsequently, the spectrogram is inputted into the CNN model.

## Training Process

The training process involves collecting a dataset of audio samples containing the keywords you want to spot. You will then use TensorFlow or similar tools to train a TinyML model. The trained model will be loaded onto the Arduino Nano 33 BLE Sense.

## Analysis and Visualization

Once the model is trained, it is important to analyze its performance and visualize the training and validation results.

<div class="image">
  <img class=center src="https://github.com/Wings-hub/ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense/blob/main/report/Images/Results/trainingProgress.png" alt="" width="500px">
  <div class="caption">Figure 2: Accuracy and loss trends over epochs</div>
</div>

## Deployment

The deployment process includes loading the trained model onto the Arduino Nano 33 BLE Sense and configuring it to respond when it recognizes one of the trained keywords.

## Usage

1. Upload the sketch to the Arduino Nano 33 BLE Sense.
2. Provide your trained model and dataset.
3. Configure the sketch to use your model.
4. Power on the Arduino Nano 33 BLE Sense and observe the RGB LED blinking when it senses a trained keyword.


## Results

The CNN model achieved a training accuracy of 89.39% with a loss of 0.32. Validation accuracy is 86% with a loss of 0.42. Test data accuracy is 85%. To improve generalization, consider tuning, regularization, or adjusting the model architecture for better performance on unseen data.

<div class="image">
  <img class=center src="https://github.com/Wings-hub/ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense/blob/main/report/Images/Results/ArduinoGreen.jpg" alt="" width="500px">
  <div class="caption">Figure 3: The board's LED response to the keyword "yes"</div>
</div>

The board's LED responses:

* Green: Reesponse to the "yes" keyword.
* Red: Reesponse to the "No" keyword.
* Blue: Reesponse to an unknown keyword.

The board relatively accurate responses when the speaker utters the keyword within a 20cm range. However, occasional issues may arise, such as failing to register a response, misclassifying "yes" or "no" as unfamiliar terms (indicated by a blue LED), or exhibiting unknown keyword responses due to environmental noise.


