from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix

from reader import Reader
from scaler import Scaler

seed = 7
np.random.seed(seed)
reader = Reader()
scaler = Scaler()

data = reader.read_dataset('Dataset/training_gold.csv')
X = reader.get_features(data, name='tempo')
y = reader.get_label(data, name='genre')

X = scaler.normalize(X)
y_categorical = scaler.categorical_binary(y)

# Creating a Neural Networks Model
model = Sequential()
model.add(Dense(28, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compiling Neural Networks Model
model.compile(loss='categorical_crossentropy', 
              optimizer="adam",
              metrics=["acc"])


X_trainMatrix = scaler.transform(X)
results = model.fit(X_trainMatrix, y_categorical, 
                    epochs=20, batch_size=128)

plt.title('Loss')
plt.plot(results.history['loss'])
plt.show()

plt.title('Accuracy')
plt.plot(results.history['acc'])
plt.show()


classic_data = pd.read_csv('Dataset/classical.csv')

print(classic_data.groupby('genre').size())

X_classic = classic_data.ix[:, 'tempo':]
y_classic = classic_data['genre']

feat_classic = list(X_classic)
for n in feat_classic:
    X_classic[n] = (X_classic[n]-np.min(X_classic[n]))/(np.max(X_classic[n])-np.min(X_classic[n]))

pred_classic = model.predict(X_classic.as_matrix())

classic_pred = []
for row in pred_classic:
    max_value, max_index = 0, 0
    for i in range(len(row)):
        if row[i] > max_value:
            max_index = i
            max_value = row[i]
            
    classic_pred.append(max_index)

print(accuracy_score(y_classic, classic_pred))

# Segmented Data
segment_data = pd.read_csv('Dataset/segment.csv')
X_segment = segment_data.ix[:, 'tempo':]

feat_segment = list(X_segment)
for n in feat_segment:
    X_segment[n] = (X_segment[n]-np.min(X_segment[n]))/(np.max(X_segment[n])-np.min(X_segment[n]))

pred_segment = model.predict(X_segment.as_matrix())

segment_total, segment_pred = [], []
for row in pred_segment:
    max_value, max_index = 0, 0
    for i in range(len(row)):
        if row[i] > max_value:
            max_index = i
            max_value = row[i]
            
    segment_total.append(max_index)
    
sns.heatmap(confusion_matrix(y_classic, classic_pred), annot=True, fmt="d")
plt.show()

for i in range(0, len(segment_total), 5):
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    count[segment_total[i]] = count[segment_total[i]] + 1
    count[segment_total[i+1]] = count[segment_total[i+1]] + 1
    count[segment_total[i+2]] = count[segment_total[i+2]] + 1
    count[segment_total[i+3]] = count[segment_total[i+3]] + 1
    count[segment_total[i+4]] = count[segment_total[i+4]] + 1
    
    max_value, max_index = 0, 0
    for j in range(len(count)):
        if count[j] > max_value:
            max_index = j
            max_value = count[j]
        
    segment_pred.append(max_index)

print(accuracy_score(y_classic, segment_pred))
sns.heatmap(confusion_matrix(y_classic, segment_pred), annot=True, fmt="d")
plt.show()
