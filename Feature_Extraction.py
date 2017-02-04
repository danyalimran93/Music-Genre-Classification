import csv
import scipy
import librosa
import numpy as np
from os import listdir
from os.path import isfile, join

def train_data(path):
    # Writter for CSV File
    writtenfile = open('bluesdata.csv', 'wb')
    fieldnames = ('id', 'songname', 'duration', 'genre', 'tempo', 'beat frames', 'beat frames sum',
                  'beat frames mean', 'beat frames variance', 'beat frames standard deviation',
                  'cent', 'cent mean', 'cent variance', 'cent standard deviation', 'bandwidth',
                  'bandwidth mean', 'bandwidth variance', 'bandwidth standard deviation', 'rolloff',
                  'rolloff mean', 'rolloff variance', 'rolloff standard deviation', 'zcr', 'zcr sum', 
                  'zcr mean', 'rmse', 'rmse mean', 'energy', 'mfcc', 'mfcc standard deviation', 'mfcc coeffecient sum',
                  'mfcc coeffecient mean', 'mfcc coeffecient variance', 'mfcc coeffecient standard deviation')
    writer = csv.DictWriter(writtenfile, fieldnames=fieldnames)
    writer.writeheader()
        
    # Feature Extraction
    song_id = 0
    file_data = [f for f in listdir(path) if isfile (join(path, f))]
    for line in file_data:
        if ( line[-1:] == '\n' ):
            line = line[:-1]

        song_name = line
        song_id = song_id + 1
        line = path + line
        
        # Librosa API for feature extraction
        y, sr = librosa.load(line, duration=60)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        
        duration = librosa.get_duration(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)

        sum_list = []
        mean_list = []
        variance_list = []
        std_dev_list = []

        for lists in mfcc:
            sum_list.append(sum(lists))
            mean_list.append(np.mean(lists))
            variance_list.append(np.var(lists))
            std_dev_list.append(np.std(lists))

        # Writting to CSV File
        writer.writerow({ 'id': song_id,
			         'songname': song_name,
                          'duration': duration,
                          'genre': 0,
                          'tempo': tempo,
                          'beat frames': beat_frames,
                          'beat frames sum': beat_frames.sum(),
                          'beat frames mean': np.mean(beat_frames),
                          'beat frames variance': np.var(beat_frames),
                          'beat frames standard deviation': np.std(beat_frames),
                          'cent': cent,
                          'cent mean': np.mean(cent),
                          'cent variance': np.var(cent),
                          'cent standard deviation': np.std(cent),
                          'bandwidth': spec_bw,
                          'bandwidth mean': np.mean(spec_bw),
                          'bandwidth variance': np.var(spec_bw),
                          'bandwidth standard deviation': np.std(spec_bw),
                          'rolloff': rolloff,
                          'rolloff mean': np.mean(rolloff),
                          'rolloff variance': np.var(rolloff),
                          'rolloff standard deviation': np.std(rolloff),
				   'zcr': zcr,
                          'zcr sum': zcr.sum(),
                          'zcr mean': np.mean(zcr),
                          'rmse': rmse,
                          'rmse mean': np.mean(rmse),
                          'energy': scipy.linalg.norm(y),
                          'mfcc': mfcc,
                          'mfcc standard deviation': np.std(mfcc),
                          'mfcc coeffecient sum': sum_list,
                          'mfcc coeffecient mean': mean_list,
                          'mfcc coeffecient variance': variance_list,
                          'mfcc coeffecient standard deviation': std_dev_list,
                       })
        
        
        print song_name
			
train_data('Blues/')