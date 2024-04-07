from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import os
import librosa
import numpy as np

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///audio_emotion.db'
db = SQLAlchemy(app)

# Define the SQLAlchemy model for the audio files
class AudioFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100))
    emotion = db.Column(db.String(50))

# Define a function to extract features from audio files
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return mfccs

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            features = extract_features(file_path)
            # Use your emotion recognition model to predict the emotion
            emotion = "Predicted Emotion"
            # Save the audio file and predicted emotion to the database
            new_audio = AudioFile(filename=filename, emotion=emotion)
            db.session.add(new_audio)
            db.session.commit()
            return 'File uploaded successfully'
    return render_template('upload.html')

@app.route('/audio_list')
def audio_list():
    audio_files = AudioFile.query.all()
    return render_template('audio_list.html', audio_files=audio_files)

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
