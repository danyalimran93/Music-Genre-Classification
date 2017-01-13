# Spectral Centroid

### Definition

The spectral centroid is a measure used in digital signal processing to characterise a spectrum. It indicates where the "center of mass" of the spectrum is. Perceptually, it has a robust connection with the impression of "brightness" of a sound. [1]

### Librosa Function 

librosa.feature.spectral_centroid(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, freq=None) [2]

### Code Template 

```Python
>>> y, sr = librosa.load(librosa.util.example_audio_file())
>>> cent = librosa.feature.spectral_centroid(y=y, sr=sr)
>>> cent
array([[ 4382.894,   626.588, ...,  5037.07 ,  5413.398]])
```

### Visual Representation

- Matplotlib.pyplot Representation of RMSE

![SC](http://librosa.github.io/librosa/_images/librosa-feature-spectral_centroid-1.png)

### References

[1] https://en.wikipedia.org/wiki/Spectral_centroid

[2] http://librosa.github.io/librosa/generated/librosa.feature.spectral_centroid.html#librosa.feature.spectral_centroid