import librosa
import numpy as np
import matplotlib.pyplot as plt

# Reading Audio File
y, sr = librosa.load(librosa.util.example_audio_file())

# Extracting RMSE Feature
S, phase = librosa.magphase(librosa.stft(y))
rmse = librosa.feature.rmse(S=S)

# Plotting RMSE Feature
plt.figure()
plt.subplot(2, 1, 1)
plt.semilogy(rmse.T, label='RMS Energy')
plt.xticks([])
plt.xlim([0, rmse.shape[-1]])
plt.legend(loc='best')
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.max), y_axis='log', x_axis='time')
plt.title('log Power spectrogram')
plt.tight_layout()
plt.show()