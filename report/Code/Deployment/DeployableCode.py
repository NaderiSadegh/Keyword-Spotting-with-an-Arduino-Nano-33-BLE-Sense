
############################
# Author: Malik Al Ashter Ghansletwala
# Date added: 15.12.2023
# Path: ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense\report\Code\Deployment\DeployableCode.py
# Version: 2
# Reviewed by:
# Review Date:
############################
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import librosa
import os

def extract_mfcc(audio_path, max_pad_len=100):
    audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

data_dir = 'path/to/training_data'
keywords = ['on', 'off', 'stop', 'go']

X, y = [], []

for keyword in keywords:
    keyword_dir = os.path.join(data_dir, keyword)
    for filename in os.listdir(keyword_dir):
        audio_path = os.path.join(keyword_dir, filename)
        mfccs = extract_mfcc(audio_path)
        X.append(mfccs.flatten())
        y.append(keyword)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the trained model (for deployment on Arduino Nano)
import joblib
joblib.dump(classifier, 'keyword_recognition_model.joblib')
