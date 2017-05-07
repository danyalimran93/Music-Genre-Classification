import pandas as pd


class Reader:
    def __init__(self):
        self.data = None
        
    def read_dataset(self, path):
        self.data = pd.read_csv(path)
        return self.data
    
    def get_features(self, data, index=None, name=None):
        if index==None:
            return data.ix[:, name:]
        else:
            return data.ix[:, index:]
        
    def get_label(self, data, index=None, name=None):
        if index==None:
            return data[name]
        else:
            return data.ix[:, index]
        
    