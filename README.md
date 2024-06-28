# Gender-Detection-through-Voice

This Python script demonstrates how to predict gender from voice recordings using a Random Forest classifier. It utilizes the `pyaudio`, `wave`, `librosa`, `numpy`, `pandas`, and `scikit-learn` libraries.

## Requirements

- Python 3.x
- PyAudio
- Librosa
- NumPy
- Pandas
- Scikit-learn

You can install the required packages using pip:

```
pip install pyaudio librosa numpy pandas scikit-learn
```

## Usage

1. **Install Dependencies**: Ensure you have all the required packages installed as mentioned above.
   
2. **Dataset**: The script assumes that the dataset is stored in a file named `voice.csv`. Ensure that your dataset is in the correct format.

3. **Run the Script**: Execute the script by running the Python file `gender_prediction.py`.

```
python gender_prediction.py
```

4. **Recording**: The script will record a 3-second audio clip. Press any key to start recording and wait until the recording finishes.

5. **Prediction**: After recording, the script will predict the gender based on the audio clip.

6. **Output**: The predicted gender (male or female) will be displayed in the console.

7. **Clean-up**: The recorded audio file (`recorded_audio.wav`) will be deleted automatically after prediction.

## Customization

- You can customize the duration of the audio recording by modifying the `duration` parameter in the `record_audio()` function.
- Adjust the threshold for gender prediction by modifying the `threshold` parameter in the `predict_gender()` function.


## Acknowledgments

- This project is inspired by similar projects in the field of machine learning and audio processing.

©Deependra Kumar 
---
