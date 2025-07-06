from flask import Flask, request, jsonify
import numpy as np
import librosa
import joblib
import os
import soundfile as sf
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained model and scaler
model = joblib.load("gender_model_best.pkl")
scaler = joblib.load("scaler.pkl")

# Function to extract MFCC features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean.reshape(1, -1)
    except Exception as e:
        print("Feature extraction error:", e)
        return None

@app.route('/predict-gender', methods=['POST'])
def predict_gender():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save the uploaded file
    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(file_path)

    # Extract features
    features = extract_features(file_path)
    if features is None:
        return jsonify({'error': 'Could not extract features from audio'}), 500

    # Scale and predict
    features_scaled = scaler.transform(features)
    probs = model.predict_proba(features_scaled)[0]
    label = int(np.argmax(probs))

    # Delete file after processing
    os.remove(file_path)

    return jsonify({
        'prediction': 'male' if label == 0 else 'female',
        'probability_male': round(float(probs[0]), 4),
        'probability_female': round(float(probs[1]), 4)
    })

if __name__ == '__main__':
    app.run(debug=True)