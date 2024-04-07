# Ensure all necessary imports are present
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from librosa import load, feature
import numpy as np
import os
import sqlite3

# Load the emotion recognition model
model = load_model("C:/Users/singh/Downloads/speech_emotion_recognition.h5")

import sqlite3
conn = sqlite3.connect('audio_emotion.db')
cur = conn.cursor()

cur.execute('''CREATE TABLE IF NOT EXISTS audio_emotion
               (id INTEGER PRIMARY KEY AUTOINCREMENT,
               filename TEXT,
               emotion TEXT)''')
conn.commit()


uploads_dir = 'uploads'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)


def extract_mfcc(audio_path):
    y, sr = load(audio_path)
    mfccs = feature.mfcc(y=y, sr=sr)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled


app = Flask(__name__)

# Set path for uploading audio files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def predict_emotion(file):
    features = extract_mfcc(file)
    features = np.expand_dims(features, axis=0)  
    features = np.expand_dims(features, axis=-1)  
    emotion_probabilities = model.predict(features)
    emotion_label = np.argmax(emotion_probabilities)
    emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad']
    predicted_emotion = emotions[emotion_label]
    return predicted_emotion


app = Flask(__name__)

# Set the path for uploading audio files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the function to predict emotion from an uploaded audio file
def predict_emotion(file):
    features = extract_mfcc(file)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)
    emotion_probabilities = model.predict(features)
    emotion_label = np.argmax(emotion_probabilities)
    emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad']
    predicted_emotion = emotions[emotion_label]
    return predicted_emotion


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            emotion = predict_emotion(file_path)

            # Store file name and predicted emotion in the database
            cur.execute("INSERT INTO audio_emotion (filename, emotion) VALUES (?, ?)", (filename, emotion))
            conn.commit()

            return f'Emotion predicted: {emotion}'
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

