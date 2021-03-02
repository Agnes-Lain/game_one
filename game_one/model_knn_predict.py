import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
from game_one.model_knn_preprocess import MyCustomPreprocessorKnn


class PredictKnn():

    def __init__(self):
        self.preprocessor = MyCustomPreprocessorKnn()
        self.load_model()

    def get_X(self, user_dict):
        return self.preprocessor.get_X_vector(user_dict)

    def load_model(self):
        with open(r"knn_model.pickle", "rb") as input_file:
            self.neigh = pickle.load(input_file)

    def get_user_id(self, user_dict):
        X = self.get_X(user_dict)
        y = self.neigh.kneighbors(X.reshape(-1, 1).transpose(), 1, return_distance=False)
        user_id = y[0][0]
        return user_id
