from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from game_one.get_meta_matrix import GetMetadata, COLUMNS
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pickle5 as pickle
import pdb


class ContentBasePred(object):

    def __init__(self):
        self.games = None
        self.metadata = None
        self.model_tf = None
        self.matrix_tf = None
        self.model_svd = None
        self.latent_df = None
        self.get_metadata()
        self.load_tf()
        self.svd_train()
        self.save_content_model()

    def get_metadata(self, game_id=None):
        '''this function will get the metadata from the get_meta_matrix class,
        and stock the data in the class constance self.data
        '''
        get_data = GetMetadata(COLUMNS)
        if game_id == None:
            get_data.get_games()
            self.games = get_data.df
            self.metadata = get_data.create_metadata()
        else:
            return get_data.create_metadata(game_id)

    def load_tf(self):
        '''This function will run the Tfid model to transfer the meta string
        into a matrix, and stock into class constance self.meta_tf
        '''
        tf_idf_vectorizer = TfidfVectorizer(stop_words='english', min_df=0.005)
        self.model_tf = tf_idf_vectorizer.fit(self.metadata)
        matrix = tf_idf_vectorizer.transform(self.metadata)
        self.matrix_tf = pd.DataFrame(matrix.toarray(), index=self.games.game_id.tolist())

    def svd_train(self):
        ''' This function will initate a SVD model to reduce the features
        '''
        svd = TruncatedSVD(n_components=200)
        self.model_svd = svd.fit(self.matrix_tf)
        latent_df = self.model_svd.transform(self.matrix_tf)
        self.latent_df = pd.DataFrame(latent_df, index=self.games.game_id.tolist())

    def save_content_model(self):
        with open('content_base_svd.pickle', 'wb') as preproc:
            pickle.dump(self, preproc)

if __name__ == '__main__':
    print ('init an instance of the class ContentBasePred')
    cbp = ContentBasePred()
    print(cbp)
    cbp.get_metadata()
    print(cbp.metadata[0])
    cbp.load_tf()
    cbp.svd_train()
    print(cbp.model_svd)
    with open('content_base_svd.pickle', 'wb') as f:
        pickle.dump(cbp, f)
    print('model saved')
