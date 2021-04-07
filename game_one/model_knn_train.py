import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle5 as pickle
from game_one.model_knn_preprocess import MyCustomPreprocessorKnn


class TrainKnn():
    ''' This class is mean to train the knn model with the matrix data from
    preprocessor, and we stock the trained model into a pickle file to load
    '''

    def __init__(self):
        self.preprocessor = self.load_prepoc_knn()
        self.neigh = None
        self.train_model()

    def load_prepoc_knn(self):
        with open('preproc.pickle', 'rb') as preproc:
            preproc = pickle.load(preproc)
            return preproc

    def save_model(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self.neigh, handle)

    def train_model(self):
        df = self.preprocessor.rating_matrix
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(df)
        self.neigh = neigh
        self.save_model('knn_model.pickle')

if __name__ == "__main__":
    knn_train = TrainKnn()
    print('knn model has been trained, ready for predict')
