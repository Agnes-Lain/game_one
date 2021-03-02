import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
from game_one.model_knn_preprocess import MyCustomPreprocessorKnn


class TrainKnn():

    def __init__(self):
        self.preprocessor = MyCustomPreprocessorKnn()
        self.neigh = None

    def train_model(self):
        df = self.preprocessor.get_rating_matrix()
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(df)
        self.neigh = neigh

    def save_model(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self.neigh, handle, protocol=pickle.HIGHEST_PROTOCOL)
