import numpy as np
import pandas as pd
import difflib
import pickle

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

class CFMTrain():

    def __init__(self):
        self.data = None

    def load_data(self):
        '''
        Get user game data
        '''
        self.data = pd.read_csv("raw_data/rawg_user_games_old.csv")
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
        svd = SVD(verbose=True, n_epochs=30, n_factors=100)
        trainset = self.surprise_data.build_full_trainset()
        svd.fit(trainset)
        self.model = svd
        return self.model

    def validate_svd_model(self, k_factors):
        '''
        SVD model 
        '''
        svd = SVD(verbose=False, n_epochs=30, n_factors=k_factors)
        cross_validate(svd, self.surprise_data, measures=['fcp','mae','rmse','mse'], cv=10, verbose=True)

    def save_model(self):
        with open('model-cfm.pickle', 'wb') as model:
            pickle.dump(self, model)
