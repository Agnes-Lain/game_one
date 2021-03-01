import pandas as pd
from sklearn.neighbors import NearestNeighbors


class KnnModel():
    def __init__(self, user_dict):
        self.user_dict = user_dict


def load_data():
    user_game_df = pd.read_csv("../raw_data/rawg_user_games.csv")
    return user_game_df


def get_ratings(df):
    filter_df = df[df['user_rating'] > 0]
    ratings_df = filter_df[['user_id', 'game_id', 'user_rating']]
    return ratings_df


def get_rating_matrix(df):
    rating_matrix = df.pivot(index='user_id', columns='game_id', values='user_rating').fillna(0)
    return rating_matrix


def get_X_matrix(matrix):
    game_id_matrix = matrix.columns
    X_matrix = pd.DataFrame(game_id_matrix)
    X_matrix['ratings'] = 0
    X_matrix = X_matrix.set_index('game_id')
    return X_matrix


def get_X_vector(X_matrix, user_dict):
    for game in user_dict:
        game_id = game["game_id"]
        ratings = game["user_rating"]
        X_matrix.loc[game_id, 'ratings'] = ratings
        X = X_matrix['ratings'].values
        return X


def train_model(df):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(df)
    return neigh

#function save model pickel


def get_user_id(neigh, X):
    y = neigh.kneighbors(X.reshape(-1, 1).transpose(), 1, return_distance=False)
    user_id = y[0][0]
    return user_id


if __name__ == '__main__':
    user_dict = [
      {
        "game_id": 26,
        "user_rating": 4
      },
      {
        "game_id": 28,
        "user_rating": 5
      },
      {
        "game_id": 30,
        "user_rating": 0
      }
    ]
    trainer = KnnModel()
    data = trainer.load_data()
    rating_df = trainer.get_ratings(data)
    rating_matrix = trainer.get_rating_matrix(rating_df)
    X_matrix = trainer.get_X_matrix(rating_matrix)
    X = trainer.get_X_vector(X_matrix, user_dict)
    model = trainer.train_model(rating_matrix)
    #save pickle user_id
    user_id = trainer.get_user_id(model, X)
