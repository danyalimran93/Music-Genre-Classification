# Mel Frequency Cepstral Coeffecient

### Definition

In sound processing, the mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency.

Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC. They are derived from a type of cepstral representation of the audio clip (a nonlinear "spectrum-of-a-spectrum"). The difference between the cepstrum and the mel-frequency cepstrum is that in the MFC, the frequency bands are equally spaced on the mel scale, which approximates the human auditory system's response more closely than the linearly-spaced frequency bands used in the normal cepstrum. This frequency warping can allow for better representation of sound, for example, in audio compression. [1]


### Librosa Function 

librosa.feature.mfcc(y=None, sr=22050, S=None, n_mfcc=20, **kwargs) [2]

### Code Template 

```Python
>>> y, sr = librosa.load(librosa.util.example_audio_file())
>>> librosa.feature.mfcc(y=y, sr=sr)
array([[ -5.229e+02,  -4.944e+02, ...,  -5.229e+02,  -5.229e+02],
       [  7.105e-15,   3.787e+01, ...,  -7.105e-15,  -7.105e-15],
       ...,
       [  1.066e-14,  -7.500e+00, ...,   1.421e-14,   1.421e-14],
       [  3.109e-14,  -5.058e+00, ...,   2.931e-14,   2.931e-14]])
```
 
### Visual Representation

- Matplotlib.pyplot Representation of MFCC

![MFCC](http://librosa.github.io/librosa/_images/librosa-feature-mfcc-1.png)

### References

[1] https://en.wikipedia.org/wiki/Mel-frequency_cepstrum

[2] http://librosa.github.io/librosa/generated/librosa.feature.mfcc.html#librosa.feature.mfcc
