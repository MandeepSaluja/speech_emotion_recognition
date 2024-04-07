import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import librosa  #to load and extract features from audio files
import librosa.display #displaying audio-related data such as waveforms and spectrograms
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')


audio_path = []
audio_label = []
for dirname, _, filenames in os.walk("C:/Users/singh/Downloads/archive (3)/TESS Toronto emotional speech set data"):
    for filename in filenames:
        audio_path.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        audio_label.append(label.lower())
    if len(audio_path) == 2800:
        break


audio_path[:5]

df = pd.DataFrame()
df['audio'] = audio_path
df['label'] = audio_label
df.head()


df['label'].value_counts()

sns.countplot(data=df, x='label')


#for creating waveform plot of audio data
def waveform_plot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()
    
def spectogram_plot(data, sr, emotion):
    x = librosa.stft(data) #Compute Short-Time Fourier Transform (STFT) of the audio data
    xdb = librosa.amplitude_to_db(abs(x)) # Convert amplitude to decibels

    plt.figure(figsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz') #Display a spectrogram of the audio data
    plt.colorbar()



%pip install --upgrade numba


emotion = 'fear'
path = np.array(df['audio'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveform_plot(data, sampling_rate, emotion)
spectogram_plot(data, sampling_rate, emotion)
Audio(path)

emotion = 'angry'
path = np.array(df['audio'][df['label']==emotion])[1]
data, sampling_rate = librosa.load(path)
waveform_plot(data, sampling_rate, emotion)
spectogram_plot(data, sampling_rate, emotion)
Audio(path)

emotion = 'disgust'
path = np.array(df['audio'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveform_plot(data, sampling_rate, emotion)
spectogram_plot(data, sampling_rate, emotion)
Audio(path)

emotion = 'neutral'
path = np.array(df['audio'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveform_plot(data, sampling_rate, emotion)
spectogram_plot(data, sampling_rate, emotion)
Audio(path)

emotion = 'sad'
path = np.array(df['audio'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveform_plot(data, sampling_rate, emotion)
spectogram_plot(data, sampling_rate, emotion)
Audio(path)

emotion = 'ps'
path = np.array(df['audio'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveform_plot(data, sampling_rate, emotion)
spectogram_plot(data, sampling_rate, emotion)
Audio(path)

emotion = 'happy'
path = np.array(df['audio'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveform_plot(data, sampling_rate, emotion)
spectogram_plot(data, sampling_rate, emotion)
Audio(path)




def extract_mfcc(file):
    y, sr = librosa.load(file, duration=3, offset=0.2)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


extract_mfcc(df['audio'][0])

X_mfcc = df['audio'].apply(lambda x: extract_mfcc(x))
X_mfcc

X = [x for x in X_mfcc]
X = np.array(X)
X = np.expand_dims(X, -1)
X.shape

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(df[['label']])
y = y.toarray()
y.shape


