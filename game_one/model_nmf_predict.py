import numpy as np
import pandas as pd
import difflib
import pickle

from surprise import NMF
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

class NMFPredict():

    def __init__(self, name):
        self.model = self.load_model(name)

    def load_model(self, name):
        with open(name, 'rb') as model:
            model = pickle.load(model)
            return model

    def all_prediction_for_a_user(self, user_id):
        '''
        Gets all the rating prediction for user.
        '''
        predictions = []
        games = list(self.model.metadata['game_id'].unique())
        for game in games:
            prediction = self.model.model.predict(uid=user_id, iid=game)
            predictions.append(prediction)
        return predictions

    def get_game_id(self, game_name):
        """
        Gets the game ID for a game_name based on the closest match in the metadata dataframe.
        """
        existing_titles = list(self.model.metadata['game_name'].values)
        closest_titles = difflib.get_close_matches(game_name, existing_titles)
        game_id = self.model.metadata[self.model.metadata['game_name'] == closest_titles[0]]['game_id'].values[0]
        return game_id

    def get_game_info(self, game_id):
        """
        Returns some basic information about a game given the game_id and the metadata dataframe.
        """
        game_info = self.model.metadata[self.model.metadata['game_id'] == game_id][['game_id','game_name', 'released', 'metacritic', 'rawg_rating']]
        return game_info.to_dict(orient='records')

    def predict_rating(self, user_id, game_name):
        """
        Predicts the user_rating (on a scale of 0-5) that a user would assign to a specific game.
        """
        game_id = self.get_game_id(game_name)
        review_prediction = self.model.model.predict(uid=user_id, iid=game_id)
        return review_prediction

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
        self.model.metadata.reset_index
        pred_user_filtered_w_n = pred_user_filtered.merge(self.model.metadata, left_on='game_id', right_on='game_id', how='left')
        data_filter_on_user = self.model.data[self.model.data['user_id'] == user_id]
        pred_user_filtered_w_n_o_n = pred_user_filtered_w_n.merge(data_filter_on_user, left_on='game_id', right_on='game_id', how='left')
        pred_user_filtered_w_n_o_n = pred_user_filtered_w_n_o_n.drop(columns=['user_id'])
        pred_user_filtered_w_n_o_n = pred_user_filtered_w_n_o_n[pred_user_filtered_w_n_o_n['purchase'] != 1]
        filtered_pred = pred_user_filtered_w_n_o_n[['game_id', 'pred_r']].head(9)
        return filtered_pred.to_dict()

    # def generate_all_prediction(self):
    #     all_preds = pd.DataFrame(self.all_prediction_for_a_user(list(self.model.data['user_id'].unique())[0])).reset_index(drop=True)
    #     for user in list(self.model.data['user_id'].unique())[1:]:
    #         pred_user = pd.DataFrame(self.all_prediction_for_a_user(user))
    #         # print(pred_user)
    #         # print(all_preds)
    #         all_preds = pd.concat([all_preds, pred_user], axis=0)
    #         # print(all_preds)
    #     return all_preds
