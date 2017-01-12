# Chromagram 

### Definition

In the music context, the term chroma feature or chromagram closely relates to the twelve different pitch classes. Chroma-based features, which are also referred to pitch class profiles, are a powerful tool for analyzing music whose pitches can be meaningfully categorized (often into twelve categories) and whose tuning approximates to the equal-tempered scale. One main property of chroma features is that they capture harmonic and melodic characteristics of music, while being robust to changes in timbre and instrumentation. 

Assuming the equal-tempered scale, one considers twelve chroma values represented by the set

{C, C♯, D, D♯, E ,F, F♯, G, G♯, A, A♯, B} [1]


### Librosa Function 

librosa.feature.chroma_stft(y=None, sr=22050, S=None, norm=inf, n_fft=2048, hop_length=512, tuning=None, **kwargs) [2]

### Visual Representation

- Matplotlib.pyplot Representation of Chromagram STFT 

![Chromagram STFT](http://librosa.github.io/librosa/_images/librosa-feature-chroma_stft-1.png)

- Complete Feature Scale

![Chromgram Complete](https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/ChromaFeatureCmajorScaleScoreAudioColor.png/685px-ChromaFeatureCmajorScaleScoreAudioColor.png)

(a). Musical Score of C-Major Scale
(b). Chromagram Obtained from the Score
(c). Audio Recording of C-Major Scaled Played on a Piano
(d). Chromgram Obtained from an Audio Recording

# References

[1] https://en.wikipedia.org/wiki/Chroma_feature

[2] http://librosa.github.io/librosa/generated/librosa.feature.chroma_stft.html#librosa.feature.chroma_stft
