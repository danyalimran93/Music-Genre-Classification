import librosa
import numpy as np
import matplotlib.pyplot as plt

# Reading Audio File
y, sr = librosa.load(librosa.util.example_audio_file())

# Extracting SC Feature
S, phase = librosa.magphase(librosa.stft(y=y))
cent = librosa.feature.spectral_centroid(S=S)

# Plotting SC Feature
plt.figure()
plt.subplot(2, 1, 1)
plt.semilogy(cent.T, label='Spectral centroid')
plt.ylabel('Hz')
plt.xticks([])
plt.xlim([0, cent.shape[-1]])
plt.legend()
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.max), y_axis='log', x_axis='time')
plt.title('log Power spectrogram')
plt.tight_layout()
plt.show()