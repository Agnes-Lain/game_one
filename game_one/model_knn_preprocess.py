import pandas as pd


class MyCustomPreprocessorKnn():

    def __init__(self):
        self.user_game_df = None
        self.ratings_df = None
        self.rating_matrix = None
        self.X_matrix = None
        self.X = None
        self.load_data()
        self.get_ratings()
        self.get_rating_matrix()
        self.get_X_matrix()

    def load_data(self):
        self.user_game_df = pd.read_csv("raw_data/rawg_user_games.csv")

    def get_ratings(self):
        filter_df = self.user_game_df[self.user_game_df['user_rating'] > 0]
        self.ratings_df = filter_df[['user_id', 'game_id', 'user_rating']]
        return self.ratings_df

    def get_rating_matrix(self):
        self.rating_matrix = self.ratings_df.pivot(index='user_id', columns='game_id', values='user_rating').fillna(0)
        return self.rating_matrix

    def get_X_matrix(self):
        game_id_matrix = self.rating_matrix.columns
        self.X_matrix = pd.DataFrame(game_id_matrix)
        self.X_matrix['ratings'] = 0
        self.X_matrix = self.X_matrix.set_index('game_id')
        return self.X_matrix

    def get_X_vector(self, user_dict):
        for game in user_dict:
            game_id = game["game_id"]
            ratings = game["user_rating"]
            self.X_matrix.loc[game_id, 'ratings'] = ratings
            X = self.X_matrix['ratings'].values
            return X
