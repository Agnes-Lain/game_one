from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle5 as pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from game_one.model_knn_predict import PredictKnn
from game_one.model_cfm_predict import CFMPredict

from typing import List, Dict
from pydantic import BaseModel

class UserDict(BaseModel):
    games: List[Dict]


app = FastAPI()

with open("content_base_svd.pickle", "rb") as f:
    CBP = pickle.load(f)
model_knn = PredictKnn('knn_model.pickle')

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],  # Allows all origins
                   allow_credentials=True,
                   allow_methods=["*"],  # Allows all methods
                   allow_headers=["*"],  # Allows all headers
                  )

@app.get("/pred_games")
def pred_games(game_id):
    game_id = int(game_id)
    if game_id in CBP.latent_df.index:
        v1 = np.array(CBP.latent_df.loc[game_id]).reshape(1, -1)
        sim1 = cosine_similarity(CBP.latent_df, v1).reshape(-1)
        dictDf = {'content': sim1}
        reco_df = pd.DataFrame(dictDf, index = CBP.latent_df.index)
        x = reco_df.sort_values('content', ascending=False, inplace=False)[1:22]
        y = x.reset_index().to_dict()
        return y
    else:
        game_meta = CBP.get_metadata(game_id)
        game_matrix = CBP.model_tf.transform(game_meta)
        print(game_matrix.shape)
        game_pred = CBP.model_svd.transform(game_matrix)
        v1 = CBP.latent_df.values
        sim2 = cosine_similarity(game_pred, v1).reshape(-1)
        dictDf = {'content': sim2}
        reco_df = pd.DataFrame(dictDf, index = CBP.latent_df.index)
        final = reco_df.sort_values('content', ascending=False, inplace=False)[1:16]
        return final.reset_index().to_dict()

@app.post("/user_pred_games")
def pred_games_col(user_dict: UserDict):
    print(user_dict)
    user_dict_2 = user_dict.games
    user_id = model_knn.get_user_id(user_dict_2)
    model_cfm = CFMPredict('model-cfm.joblib')
    reco =  model_cfm.generate_recommendation(user_id, 3)
    return reco

