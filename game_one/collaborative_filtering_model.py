import numpy as np
import pandas as pd

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

class CFModel():

    def __init__(self):
        self.df = None

    def load_data():
        '''
        Get user game data
        '''
        df = pd.read_csv("rawg_user_games.csv")
        return df

    def get_ratings_and_meta(df):
        filter_df = df[df['user_rating']>0]
        ratings_df = filter_df[['user_id','game_id','user_rating']]
        metadata = df[['game_id','game_name', 'released', 'metacritic', 'rawg_rating']]
        metadata['dummies'] = 0
        meta = metadata.groupby(by=['game_id','game_name', 'released', 'metacritic', 'rawg_rating']).sum().drop(columns='dummies').reset_index()
        return ratings_df, meta

    def get_surprise_data(df):
        '''
        Transform ratings_df in sparse matrix to be used by Surprize
        '''
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df, reader)
        return data

    def get_train_set(surprise_df):
        '''
        Get a trainset
        '''
        return surprise_df.build_full_trainset()

    def run_model(surprise_df):
        '''

        '''
        svd = SVD(verbose=True, n_epochs=10)
        trainset = surprise_df.build_full_trainset()
        svd.fit(trainset)
        return svd

    def validate_svd_model(surprise_df):
        '''
        SVD model 
        '''
        svd = SVD(verbose=True, n_epochs=10)
        cross_validate(svd, surprise_df, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    def all_prediction_for_a_user(user_id, model, rating_df):
        '''
        Gets all the rating prediction for user.
        '''
        predictions = []
        games = list(rating_df['game_id'].unique())
        for game in games:
            prediction = model.predict(uid=user_id, iid=game)
            predictions.append(prediction)
        return predictions

    def get_game_id(game_name, metadata):    
        """
        Gets the game ID for a game_name based on the closest match in the metadata dataframe.
        """
        existing_titles = list(metadata['game_name'].values)
        closest_titles = difflib.get_close_matches(game_name, existing_titles)
        game_id = metadata[metadata['game_name'] == closest_titles[0]]['game_id'].values[0]
        return game_id

    def get_game_info(game_id, metadata):
        """
        Returns some basic information about a game given the game_id and the metadata dataframe.
        """
        game_info = metadata[metadata['game_id'] == game_id][['game_id','game_name', 'released', 'metacritic', 'rawg_rating']]
        return game_info.to_dict(orient='records')

    def predict_rating(user_id, game_name, model, metadata):
        """
        Predicts the user_rating (on a scale of 0-5) that a user would assign to a specific game. 
        """

        game_id = get_game_id(game_name, metadata)
        review_prediction = model.predict(uid=user_id, iid=game_id)
        return review_prediction.est

    def generate_recommendation(user_id, model, rating_df, metadata, thresh):
        """
        Generates a list of games recommendation for a user based on a rating threshold. Only
        games with a predicted rating at or above the threshold will be recommended
        """
        pred_user = pd.DataFrame(all_prediction_for_a_user(user_id, model, rating_df))\
                .drop(columns=['uid','r_ui', 'details'])\
                .sort_values('est',ascending=False)
        pred_user_filtered = pred_user[pred_user['est'] >= thresh]
        pred_user_filtered.rename(columns = {'iid':'game_id', 'est':'pred_r'}, inplace = True)
        metadata.reset_index
        pred_user_filtered_w_n = pred_user_filtered.merge(metadata, left_on='game_id', right_on='game_id', how='left')
        return pred_user_filtered_w_n
