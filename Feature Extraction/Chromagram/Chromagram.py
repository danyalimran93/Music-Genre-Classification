import librosa
import matplotlib.pyplot as plt

# Reading Audio File
y, sr = librosa.load(librosa.util.example_audio_file())

# Extracting Chromagram STFT Feature
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Plotting Chromagram STFT Feature
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram STFT')
plt.tight_layout()
plt.show()