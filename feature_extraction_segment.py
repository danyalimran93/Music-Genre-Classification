import librosa
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

class FeatureExtractor:
    def __init__(self):
        self.data = None
        
    def extract(self,path):
        id = 0
        self.data, start_time = [], [0, 4, 7, 10, 13]
        file_data = [f for f in listdir(path) if isfile (join(path, f))]
        for line in file_data:
            if ( line[-1:] == '\n' ):
                line = line[:-1]
                
            
            id = id + 1
            songname = path + '/' + line
            
            print("Reading Song#{}: ".format(id) + songname)
            
            for i in range(len(start_time)):
                features = []
                y, sr = librosa.load(songname, duration=3, offset=start_time[i])
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                rmse = librosa.feature.rmse(y=y)
                cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
            
                features.append(id)
                features.append(line)
                features.append(tempo)
                features.append(np.sum(beats))
                features.append(np.mean(chroma_stft))
                features.append(np.mean(rmse))
                features.append(np.mean(cent))
                features.append(np.mean(spec_bw))
                features.append(np.mean(rolloff))
                features.append(np.mean(zcr))
                for coefficient in mfcc:
                    features.append(np.mean(coefficient))
                
                self.data.append(features)

    def get_data(self):
        return self.data
        
# main ()
np.set_printoptions(threshold=np.inf)
extractor = FeatureExtractor()
extractor.extract('Dataset/MillionSong')

pd.set_option('display.max_colwidth', -1)

heading = ['id', 'songname', 'tempo', 'beats', 'chromagram', 'rmse',
           'centroid', 'bandwidth', 'rolloff', 'zcr', 'mfcc1', 'mfcc2',
           'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
           'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
           'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']
df = pd.DataFrame(extractor.get_data(), columns=heading)
df.to_csv('sub_cliptest.csv')