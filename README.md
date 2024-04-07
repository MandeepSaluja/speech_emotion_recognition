## speech_emotion_recognition

Project Setup:



Dependencies to be installed in your system: In requirements.txt



flask #to create the web application where users can upload audio files and view the predicted emotions)


flask_sqlalchemy #Flask-SQLAlchemy is used to interact with an SQLite database to store information about uploaded audio files and their predicted emotions. 


librosa #to extract features from the uploaded audio files, which are then used as input to the emotion recognition model 


tensorflow 


#Dataset used: Toronto emotional speech set (TESS), Link: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess/data

Emotions:
Neutral:0


Angry:1


Disgust:2


Fear:3


Happy:4


Sad:5


Surprise:6



#The format of the audio file is a WAV format



