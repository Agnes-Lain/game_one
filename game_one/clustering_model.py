import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
from game_one.preprocess_knn import MyCustomPreprocessorKnn


class KnnModel():

    def __init__(self):
        self.preprocessor = MyCustomPreprocessorKnn()
        self.neigh = None

    def get_X(self, user_dict):
        return self.preprocessor.get_X_vector(user_dict)

    def train_model(self):
        df = self.preprocessor.get_rating_matrix()
        self.neigh = NearestNeighbors(n_neighbors=1)
        self.neigh.fit(df)


    def save_model(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self.neigh, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def get_user_id(self, user_dict):
        X = self.get_X(user_dict)
        y = self.neigh.kneighbors(X.reshape(-1, 1).transpose(), 1, return_distance=False)
        user_id = y[0][0]
        return user_id


if __name__ == '__main__':
    trainer = KnnModel()
    x = MyCustomPreprocessorKnn()
    x.load_data()
    trainer.train_model()
    trainer.save_model('knn_model.pickle')
    print(trainer.get_user_id([
      {
        "game_id": 26,
        "user_rating": 4
      },
      {
        "game_id": 28,
        "user_rating": 5
      }
    ]))
