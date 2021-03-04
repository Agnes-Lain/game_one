import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle5 as pickle
from game_one.model_knn_preprocess import MyCustomPreprocessorKnn


class PredictKnn():

    def __init__(self, filename):
        self.preprocessor = self.load_prepoc_knn()
        self.filename = filename
        self.load_model()

    def load_prepoc_knn(self):
        with open('preproc.pickle', 'rb') as preproc:
            preproc = pickle.load(preproc)
            return preproc

    def get_X(self, user_dict):
        return self.preprocessor.get_X_vector(user_dict)

    def load_model(self):
        with open(self.filename, "rb") as input_file:
            self.neigh = pickle.load(input_file)

    def get_user_id(self, user_dict):
        # mypreprocess = MyCustomPreprocessorKnn()
        # rating_matrix = mypreprocess.get_rating_matrix()
        rating_matrix = self.preprocessor.get_rating_matrix()
        user_index = rating_matrix.index
        print(user_index)
        X = self.get_X(user_dict)
        y = self.neigh.kneighbors(X.reshape(-1, 1).transpose(), 1, return_distance=False)
        print(y)
        user_id_index = y[0][0]
        user_id = user_index[user_id_index]
        return user_id

if __name__ == "__main__":
    knn = PredictKnn()
    print(
        knn.get_user_id(
            [{'game_id': 10646, 'user_rating': 4},
             {'game_id': 19279, 'user_rating': 5},
             {'game_id': 18099, 'user_rating': 3},
             {'game_id': 19495, 'user_rating': 4},
             {'game_id': 264828, 'user_rating': 3},
             {'game_id': 11593, 'user_rating': 5},
             {'game_id': 21371, 'user_rating': 4},
             {'game_id': 19442, 'user_rating': 4},
             {'game_id': 51487, 'user_rating': 4},
             {'game_id': 2536, 'user_rating': 4},
             {'game_id': 284, 'user_rating': 5},
             {'game_id': 3955, 'user_rating': 4},
             {'game_id': 59637, 'user_rating': 3},
             {'game_id': 19458, 'user_rating': 5},
             {'game_id': 2830, 'user_rating': 4},
             {'game_id': 1682, 'user_rating': 3},
             {'game_id': 4223, 'user_rating': 3},
             {'game_id': 39, 'user_rating': 3},
             {'game_id': 2454, 'user_rating': 5}]
        )
    )
