############################
# Original Author: TensorFlow team
# Source: https://www.tensorflow.org/lite/api_docs/python/tf/lite
# Added to the project by: Malik Al Ashter Ghansletwala
# Date added: 15.12.2023
# Path: ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense\report\Code\Deployment\TensorflowLiteSampleCode.py
# Version: 2
# Reviewed by:
# Review Date:
############################

import tensorflow as tf
import numpy as np
import micro_speech

# Load the TensorFlow Lite model.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize the microphone.
microphone = micro_speech.Microphone()

# Start the microphone recording.
microphone.start()

# Record audio for 1 second.
audio = microphone.record(1)

# Preprocess the audio data.
input_data = np.array(audio, dtype=np.float32).reshape(1, 1960)
input_data = (input_data - 128.0) / 128.0

# Run inference on the model.
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Print the recognized keyword.
if output_data[0][0] > output_data[0][1]:
    print("Keyword: on")
else:
    print("Keyword: off")
