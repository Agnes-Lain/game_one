import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle5 as pickle
from game_one.model_knn_preprocess import MyCustomPreprocessorKnn


class TrainKnn():

    def __init__(self):
        self.preprocessor = self.load_prepoc_knn()
        self.neigh = None

    def load_prepoc_knn(self):
        with open('preproc.pickle', 'rb') as preproc:
            preproc = pickle.load(preproc)
            return preproc

    def train_model(self):
        df = self.preprocessor.rating_matrix
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(df)
        self.neigh = neigh
        self.save_model('knn_model.pickle')

    def save_model(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self.neigh, handle)
