import numpy as np
import pandas as pd
import requests

class FetchUserGames(object):
    """
        with given user_id of rawg, this class will return a list of game dictionaries. Can be used as json format
        of API return results.
    """

    def __init__(self, user_id):
        self.user_id = user_id
        self.results = None
        self.url = f"https://api.rawg.io/api/users/{user_id}/games?"

    def get_user_game(self, result):
        """ create a dictionary for each parsed games from API rawg"""
        user_rating = result["user_rating"]
        # print(user_rating)

        metacritic = result["metacritic"]
        # print(metacritic)

        rawg_rating = result["rating"]
        # print(rawg_rating)

        game_release = result["released"]
        # print (game_release)

        game_id = result['id']
        # print(game_id)

        game_slut = result["slug"]
        # print(game_slut)

        game_name = result['name']
        # print(game_name)
        play_time = result['playtime']
        # print(play_time)

        user_game = {"user_id": self.user_id,
                     "game_id": game_id,
                     "game_slug": game_slut,
                     "game_name": game_name,
                     "user_rating": user_rating,
                     "metacritic": metacritic,
                     "rawg_rating": rawg_rating,
                    "released": game_release,
                    "play_time": play_time}

        return user_game


    def fetch_game(self):
        '''send a get request to the rawg API and get back the returned data'''
        response = requests.get(self.url)
        data = response.json()
        return data

    def get_results(self):
        '''This fonction will iterate the return results of json of 20 games
            and call the function get_user_game to make dictionary of each
            game, and then stock into a new list user_games
        '''
        user_games = []
        for result in self.results:
            game = self.get_user_game(result)
    #         print(game)
            user_games.append(game)
        return user_games

    def get_user_games(self):
        '''This function is main function of the class to get user all games
            from the rawg API. It's looping while there are more pages of games.
        '''
        total_user_games = []
        try:
            data = self.fetch_game()
            self.results = data['results']
            games = self.get_results()
            total_user_games.extend(games)
            while data['next'] != None:
                self.url = data['next']
                data = self.fetch_game()
                self.results = data['results']
                games = self.get_results()
                total_user_games.extend(games)
        except:
            print ("no user or no user games")

        return total_user_games

if __name__ == '__main__':
    print ('initate the instance of the class FetchUserGames')
    fetch = FetchUserGames('agnes4')
    print('fetch games of the given user_id')
    user_games = fetch.get_user_games()
    print(len(user_games))
