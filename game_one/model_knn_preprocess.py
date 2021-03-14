import pandas as pd
import pickle5 as pickle
from sklearn.metrics.pairwise import cosine_similarity


class MyCustomPreprocessorKnn():

    def __init__(self):
        # self.user_game_df = None
        # self.ratings_df = None
        self.rating_matrix = None
        self.X_matrix = None
        # self.X = None
        # self.load_data()
        # self.get_ratings()
        # self.get_rating_matrix()
        # self.get_X_matrix()
        self.save_knn_preproc()

    def load_data(self):
        user_game_df = pd.read_csv("raw_data/rawg_user_games.csv")
        return user_game_df

    def get_ratings(self, user_game_df):
        # filter_df = self.user_game_df[self.user_game_df['user_rating'] > 0]
        filter_df = user_game_df[user_game_df['user_rating'] > 0]
        ratings_df = filter_df[['user_id', 'game_id', 'user_rating']]
        return ratings_df

    def get_rating_matrix(self, ratings_df):
        self.rating_matrix = ratings_df.pivot(index='user_id', columns='game_id', values='user_rating').fillna(0)
        # return self.rating_matrix

    def get_X_matrix(self):
        game_id_matrix = self.rating_matrix.columns
        self.X_matrix = pd.DataFrame(game_id_matrix)
        self.X_matrix['ratings'] = 0
        self.X_matrix = self.X_matrix.set_index('game_id')
        # return self.X_matrix

    def load_content_base(self):
        with open('content_base_svd.pickle', 'rb') as cbp:
            cbp = pickle.load(cbp)
            return cbp

    def get_X_vector(self, user_dict):
        X_matrix_filled =self.X_matrix.copy()
        for game in user_dict:
            game_id = game["game_id"]
            ratings = game["user_rating"]
            if not(game_id in self.X_matrix.index):
                cbp = self.load_content_base()
                game_meta = cbp.get_metadata(game_id)
                game_matrix = cbp.model_tf.transform(game_meta)
                game_pred = cbp.model_svd.transform(game_matrix)
                v1 = cbp.latent_df.values
                sim2 = cosine_similarity(game_pred, v1).reshape(-1)
                dictDf = {'content': sim2}
                reco_df = pd.DataFrame(dictDf, index = cbp.latent_df.index)
                final = reco_df.sort_values('content', ascending=False, inplace=False)[1:100]
                game_ids = final.reset_index().to_dict()["index"].values()
                # print(game_ids)
                for game in game_ids:
                    games_tmp = []
                    if game in self.X_matrix.index:
                        games_tmp.append(game)
                game_id = games_tmp[0]
            X_matrix_filled.loc[game_id, 'ratings'] = ratings
        X = X_matrix_filled['ratings'].values
        return X

    def save_knn_preproc(self):
        user_game_df = self.load_data()
        ratings_df = self.get_ratings(user_game_df)
        self.get_rating_matrix(ratings_df)
        self.get_X_matrix()
        with open('preproc.pickle', 'wb') as preproc:
            pickle.dump(self, preproc)
