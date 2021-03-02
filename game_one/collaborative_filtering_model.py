import numpy as np
import pandas as pd
import difflib
import joblib

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

class CFModel():

    def __init__(self):
        self.data = None

    def load_data(self):
        '''
        Get user game data
        '''
        self.data = pd.read_csv("raw_data/rawg_user_games.csv")
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

        '''
        svd = SVD(verbose=True, n_epochs=10)
        trainset = self.surprise_data.build_full_trainset()
        svd.fit(trainset)
        self.model = svd
        return self.model

    def validate_svd_model(self):
        '''
        SVD model 
        '''
        svd = SVD(verbose=True, n_epochs=10)
        cross_validate(svd, self.surprise_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    def all_prediction_for_a_user(self, user_id):
        '''
        Gets all the rating prediction for user.
        '''
        predictions = []
        games = list(self.ratings_df['game_id'].unique())
        for game in games:
            prediction = self.model.predict(uid=user_id, iid=game)
            predictions.append(prediction)
        return predictions

    def get_game_id(self, game_name):    
        """
        Gets the game ID for a game_name based on the closest match in the metadata dataframe.
        """
        existing_titles = list(self.metadata['game_name'].values)
        closest_titles = difflib.get_close_matches(game_name, existing_titles)
        game_id = self.metadata[self.metadata['game_name'] == closest_titles[0]]['game_id'].values[0]
        return game_id

    def get_game_info(self, game_id):
        """
        Returns some basic information about a game given the game_id and the metadata dataframe.
        """
        game_info = self.metadata[self.metadata['game_id'] == game_id][['game_id','game_name', 'released', 'metacritic', 'rawg_rating']]
        return game_info.to_dict(orient='records')

    def predict_rating(self, user_id, game_name):
        """
        Predicts the user_rating (on a scale of 0-5) that a user would assign to a specific game. 
        """

        game_id = self.get_game_id(game_name)
        review_prediction = self.model.predict(uid=user_id, iid=game_id)
        return review_prediction.est

    def generate_recommendation(self, user_id, thresh):
        """
        Generates a list of games recommendation for a user based on a rating threshold. Only
        games with a predicted rating at or above the threshold will be recommended
        """
        pred_user = pd.DataFrame(self.all_prediction_for_a_user(user_id))\
                .drop(columns=['uid','r_ui', 'details'])\
                .sort_values('est',ascending=False)
        pred_user_filtered = pred_user[pred_user['est'] >= thresh]
        pred_user_filtered.rename(columns = {'iid':'game_id', 'est':'pred_r'}, inplace = True)
        self.metadata.reset_index
        pred_user_filtered_w_n = pred_user_filtered.merge(self.metadata, left_on='game_id', right_on='game_id', how='left')
        return pred_user_filtered_w_n

    def save_model(self):
        with open('model-cfm.joblib', 'wb') as model:
            joblib.dump(self, model)

    def load_model(name):
        with open(name, 'rb') as model: 
            model = joblib.load(model)
            return model