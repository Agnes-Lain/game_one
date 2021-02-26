import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


class MyMetrics():
    def __init__(self, target):
        self.target = target
        self.user_game_df = None
        self.X_train = None
        self.X_test = None

    def load_data(self, file):
        self.user_game_df = pd.read_csv(file)
        self.user_game_df = self.user_game_df[self.user_game_df['user_rating'] > 0]
        return self.user_game_df

    def train_test_split(self):
        df = self.user_game_df
        df = df[['user_id', 'game_name', self.target]]
        self.X_train, self.X_test = train_test_split(df, test_size=0.01, random_state=37)
        return self.X_train, self.X_test

    def transform_df(self):
        X_train, X_test = self.train_test_split()
        transformed_df = X_train.pivot(index='game_name', columns='user_id', values=self.target).fillna(0)
        return transformed_df

    def svd_inverse(self, nb_components):
        matrix_df = self.transform_df()
        svd = TruncatedSVD(n_components=nb_components)
        games_factors = svd.fit_transform(matrix_df)
        r = svd.inverse_transform(games_factors)
        return pd.DataFrame(r, index=matrix_df.index, columns=matrix_df.columns)

    def scale_pred_matrix(self, nb_components):
        df = self.svd_inverse(nb_components)
        scaler = MinMaxScaler()
        return pd.DataFrame(scaler.fit_transform(df), index=df.index)
        #return self.df_scaled

    def get_predicted_ratings(self, user_id, game_name):
        matrix = self.scaled_df
        try:
            pred = matrix[matrix.index == game_name][user_id][0]
        except:
            pred = 0
        return pred

    def make_y_pred(self, nb_components):
        self.scaled_df = self.scale_pred_matrix(nb_components)
        y_pred = []
        for _, row in self.X_test.iterrows():
            user_id = row['user_id']
            game_name = row['game_name']
            y_pred.append(self.get_predicted_ratings(user_id, game_name))
        self.X_test['y_pred'] = y_pred
        return self.X_test

    def mae_score(self, nb_components):
        ratings_pred = self.make_y_pred(nb_components)
        mae = mean_absolute_error(ratings_pred['user_rating'], ratings_pred['y_pred'])
        return mae
