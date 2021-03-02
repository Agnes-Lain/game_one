import numpy as np
import pandas as pd
import difflib
import pickle

from surprise import NMF
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

class NMFTrain():

    def __init__(self):
        self.data = None

    def load_data(self):
        '''
        Get user game data
        '''
        self.data = pd.read_csv("raw_data/rawg_user_games.csv")
        self.data['purchase'] = 1
        return self.data

    def get_ratings_and_meta(self):
        filter_df = self.data[self.data['user_rating']>0]
        self.ratings_df = filter_df[['user_id','game_id','user_rating']]
        meta = self.data[['game_id','game_name', 'released', 'metacritic', 'rawg_rating']]
        meta['dummies'] = 0
        self.metadata = meta.groupby(by=['game_id','game_name', 'released', 'metacritic', 'rawg_rating']).sum().drop(columns='dummies').reset_index()
        return self.ratings_df, self.metadata

    def get_surprise_data(self, min_rate=1, max_rate=5):
        '''
        Transform ratings_df in sparse matrix to be used by Surprize
        '''
        reader = Reader(rating_scale=(min_rate, max_rate))
        self.surprise_data = Dataset.load_from_df(self.ratings_df, reader)
        return self.surprise_data

    # def get_train_set(surprise_df):
    #     '''
    #     Get a trainset
    #     '''
    #     return surprise_df.build_full_trainset()

    def run_model(self):
        '''
        Create a model and train it 
        '''
        nmf = NMF(verbose=True, n_epochs=10)
        trainset = self.surprise_data.build_full_trainset()
        nmf.fit(trainset)
        self.model = nmf
        return self.model

    def validate_svd_model(self):
        '''
        SVD model 
        '''
        nmf = NMF(verbose=True, n_epochs=10)
        cross_validate(nmf, self.surprise_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    def save_model(self):
        with open('model-nmf.pickle', 'wb') as model:
            pickle.dump(self, model)
