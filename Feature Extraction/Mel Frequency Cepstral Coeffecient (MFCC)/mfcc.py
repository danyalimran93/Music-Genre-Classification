import librosa
import matplotlib.pyplot as plt

# Reading Audio File
y, sr = librosa.load(librosa.util.example_audio_file())

# Extracting MFCC Feature
mfcc = librosa.feature.mfcc(y=y, sr=sr)

# Plotting MFCC Feature
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()