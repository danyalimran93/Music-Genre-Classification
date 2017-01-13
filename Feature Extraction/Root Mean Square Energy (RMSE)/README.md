# Root Mean Square Energy

# Definition

In statistics and its applications, the root mean square (abbreviated RMS or rms) is defined as the square root of mean square (the arithmetic mean of the squares of a set of numbers). The RMS is also known as the quadratic mean and is a particular case of the generalized mean with exponent 2. RMS can also be defined for a continuously varying function in terms of an integral of the squares of the instantaneous values during a cycle. [1]


# Librosa Function 

librosa.feature.rmse(y=None, S=None, n_fft=2048, hop_length=512) [2]

### Code Template 

```Python
>>> y, sr = librosa.load(librosa.util.example_audio_file())
>>> librosa.feature.rmse(y=y)
array([[ 0.   ,  0.056, ...,  0.   ,  0.   ]], dtype=float32)
```

# Visual Representation

- Matplotlib.pyplot Representation of RMSE

![RMSE](http://librosa.github.io/librosa/_images/librosa-feature-rmse-1.png)

# References

[1] https://en.wikipedia.org/wiki/Root_mean_square

[2] http://librosa.github.io/librosa/generated/librosa.feature.rmse.html#librosa.feature.rmse