import numpy as np
import pandas as pd
import pyaudio
import wave
import os
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("voice.csv")  # Assuming the dataset is stored in a file called "voice.csv"

# Convert categorical labels into numerical labels
data['label'] = data['label'].map({'male': 0, 'female': 1})

# Extract features and labels
X = data.drop('label', axis=1)
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform grid search (optional) - comment out if not needed
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20],
# }
# grid_search = GridSearchCV(rf, param_grid, cv=5)
# grid_search.fit(X_train_scaled, y_train)

# Train the final model (with or without grid search)
# final_model = grid_search.best_estimator_ if grid_search else rf
final_model = rf.fit(X_train_scaled, y_train)

# Evaluate model performance on test set (comment out if not needed)
# y_pred = final_model.predict(X_test_scaled)
# accuracy = accuracy_score(y_test, y_pred)
# print("Test set accuracy:", accuracy)
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion matrix:")
# print(conf_matrix)

# Function to record audio
def record_audio(filename, duration=3):
    chunk = 1024  # Record in chunks
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Sample rate
    seconds = duration

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print("Recording...")

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Record for the specified duration
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print("Finished recording.")

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

# Function to extract MFCC features from audio file
def extract_features(filename):
    y, sr = librosa.load(filename, sr=None)  # Load audio file

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Using 20 MFCC coefficients
    mfccs_mean = np.mean(mfccs, axis=1)  # Calculate the mean of MFCC coefficients
    return mfccs_mean.reshape(1, -1)  # Return the feature vector as a 2D array


# Function to predict gender from voice with custom threshold
def predict_gender(filename, threshold=0.5):
    # Extract features from audio file
    features = extract_features(filename)

    # Scale the features
    features_scaled = scaler.transform(features)

    # Use the final model to predict gender and obtain probabilities
    probabilities = final_model.predict_proba(features_scaled)

    # Print predicted probabilities for both classes
    print("Predicted probabilities for male and female:", probabilities)

    # Make the final prediction based on probabilities and custom threshold
    if probabilities[0, 0] > threshold:
        return "Male"
    else:
        return "Female"


# Main function
def main():
    # Record audio
    filename = "recorded_audio.wav"
    record_audio(filename)

    # Predict gender
    gender = predict_gender(filename)
    print("Predicted gender:", gender)

    # Clean up - delete the recorded audio file
    os.remove(filename)

if __name__ == "__main__":
    main()
