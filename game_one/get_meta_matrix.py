import string
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from game_one.parse_data import get_game


COLUMNS = ['game_genres',
           'game_tags']

class GetMetadata(object):
    def __init__(self,columns):
        self.columns = columns
        self.df = None

    def get_games(self):
        self.df = pd.read_csv('raw_data/rawg_games.csv')
#         return self.df.head(1)

    def stringfy_columns(self):
#         print(self.columns)
        for column in self.columns:
            self.df[column] = self.df[column].astype(str)
#             print(type(self.df[column][0]))
#             print(self.df[column][0])

    def merge_metadata(self):
        self.df['metadata'] = ''
        for column in self.columns:
#             print(type(self.df[column][0]))
            self.df['metadata'] += (self.df[column] + ' ')
#         print(self.df['metadata'][0])
#         print(type(self.df['metadata'][0]))


    def replace_punctuations(self, text):
        punctuations = string.punctuation.replace("|", "")+'•'
        for punctuation in punctuations:
            text = text.replace(punctuation, ' ')
        text = text.replace('br', '')
        return text.lower()

    def create_metadata(self, game_id = None):
        if game_id == None:
            self.get_games()
        else:
            self.df = pd.DataFrame(get_game(game_id))
        self.stringfy_columns()
        self.merge_metadata()
#         print(self.df['metadata'][0])
        self.df['metadata'] = self.df['metadata'].apply(lambda x: self.replace_punctuations(x))
        return self.df['metadata']

if __name__ == '__main__':
    print ('Initiate the GetMetaMatrix class and store as instance')
    get_metadata = GetMetadata(COLUMNS)
    print('transfer given columns into a meta string for NPL use purpose after')
    metadata = get_metadata.create_metadata()
    print(f'{metadata.shape} games has been added the metadata')
