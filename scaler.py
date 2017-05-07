import numpy as np

from keras.utils import np_utils

class Scaler:
    @staticmethod
    def normalize(data):
        featureName = list(data)
        for name in featureName:
            data[name] = (data[name]-np.min(data[name]))/(np.max(data[name])-np.min(data[name]))
            
        return data
    
    @staticmethod
    def categorical_binary(target):
        return np_utils.to_categorical(target)
    
    @staticmethod
    def transform(data):
        return np.array(data)