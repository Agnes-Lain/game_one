import numpy as np
import pandas as pd
import requests
import math
from google.cloud import storage


def get_user(id):
    try:
        # api_key="58eeb730ca1a47e8aa9d130c3127d412"
        url=f"https://api.rawg.io/api/users/{id}"
        response = requests.get(url)
        data = response.json()
        if data["games_count"]>0:
            users.append({'id':id,
                          "games_count": data["games_count"],
                          "games_wishlist_count":data["games_wishlist_count"]})
    except:
        print ("no user or no user games")


def get_user_games(user_id, games_count):
    try:
        pages = math.ceil(games_count/20)

        for page in np.arange(1,pages+1):
            print(f'pages number is: {page}')
            url=f"https://api.rawg.io/api/users/{user_id}/games?pages={page}"
            response = requests.get(url)
            data = response.json()

            # print(len(data['results']))
            for result in data['results']:
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

                user_games.append({"user_id":user_id,
                                   "game_id": game_id,
                                   "game_slug":game_slut,
                                   "game_name":game_name,
                                   "user_rating":user_rating,
                                   "metacritic":metacritic,
                                   "rawg_rating":rawg_rating,
                                   "released":game_release,
                                   "play_time":play_time})
    except:
        print ("no user or no user games")


def get_games(game_id):
    try:
        # api_key="58eeb730ca1a47e8aa9d130c3127d412"
        url=f"https://api.rawg.io/api/games/{game_id}"
        response = requests.get(url)
        data = response.json()

        # this is to transfer the list of dict into a string to stock in one cell:
        rawg_ratings = []
        for rating in data["ratings"]:
            rawg_ratings.append(f"{rating['id']}|{rating['title']}|{rating['count']}|{rating['percent']}")
        # print(', '.join(rawg_ratings))

        # this is to get the list of platforms availble for each games:
        game_platforms = []
        for platform in data["platforms"]:
            game_platforms.append(f"{platform['platform']['id']}|{platform['platform']['name']}")

        # print(', '.join(game_platforms))

        game_genres = []
        for genre in data["genres"]:
            game_genres.append(genre['name'])
        # print(', '.join(game_genres))

        game_tags = []
        for tag in data['tags']:
            game_tags.append(tag['name'])
        # print(', '.join(game_genres))

        developers = []
        if data["developers"]:
            for developer in data["developers"]:
                developers.append(f"{developer['id']}|{developer['name']}")
        # print(', '.join(developers))

        publishers = []
        if data["publishers"]:
            for publisher in data["publishers"]:
                publishers.append(f"{publisher['id']}|{publisher['name']}")
        # print(', '.join(publishers))

        rawg_games.append({"game_id":game_id,
                           "slug": data["slug"],
                           "name":data["name"],
                           "description":data["description"],
                           "released":data["released"],
                           "rating":data["rating"],
                           "detail_ratings":', '.join(rawg_ratings),
                           "ratings_count":data["ratings_count"],
                           "suggestions_count":data["suggestions_count"],
                           "game_series_count":data["game_series_count"],
                           "reviews_count":data["reviews_count"],
                           "metacritic":data["metacritic"],
                           "game_platforms":', '.join(game_platforms),
                           "game_genres":', '.join(game_genres),
                           "game_tags":', '.join(game_tags),
                           "developers": ', '.join(developers),
                           "publishers": ', '.join(publishers)})

        print(f"game-{game_id}-{data['name']} has added to list")

    except:
        print ("no games")
