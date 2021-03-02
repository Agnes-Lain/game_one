from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



app = FastAPI()

with open("content_base_svd.pickle", "rb") as f:
    CBP = pickle.load(f)

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
        x = reco_df.sort_values('content', ascending=False, inplace=False).head(20)
        y = x.reset_index().to_dict()
        return y
    # else:
    #     game_meta = CBP.get_metadata(game_id)
    #     game_matrix = CBP.model_tf.transform(game_meta)
    #     game_pred = CBP.model_svd.transform(game_matrix)
    #     game_df = pd.DataFrame(game_pred, index=CBP.games.game_id.tolist())
    #     v2 = np.array(game_df.loc[game_id]).reshape(1, -1)
    #     sim2 = cosine_similarity(game_df, v2).reshape(-1)
    #     dictDf = {'content': sim2}
    #     reco_df = pd.DataFrame(dictDf, index = game_df.index)
    #     final = reco_df.sort_values('content', ascending=False, inplace=False).head(20)
    #     return final.reset_index().to_dict()

