import pandas as pd
from typing import List
import re



class DataPrep:
    ratings_names: List[str] = ["userId", "movieId", "rating", "timestamp"]
    movies_names: List[str] = ["movieId", "title", "genres"]
    user_df_names: List[str] = ["userId", "gender", "age", "occupation", "zip-code"]

    def __init__(self, data_path: str):
        self.ratings: pd.DataFrame = pd.read_csv(data_path+ "/ratings.dat",sep="::", names=DataPrep.ratings_names, engine='python')
        self.movies: pd.DataFrame = pd.read_csv(data_path + "/movies.dat",sep="::", names=DataPrep.movies_names, engine='python', encoding='ISO-8859-1')
        self.users: pd.DataFrame = pd.read_csv(data_path + "/users.dat",sep="::", names=DataPrep.user_df_names, engine='python')
        self.merge_data()
        
    def merge_data(self):
        merged_data: pd.DataFrame = pd.merge(pd.merge(self.ratings, self.movies, on='movieId'), self.users, on='userId')
        splitted_genres: pd.DataFrame = merged_data.genres.str.get_dummies('|')
        cleaned_data: pd.DataFrame = pd.concat([merged_data, splitted_genres], axis=1)
        # cleaned_data['zip-code'] = cleaned_data['zip-code'].astype(str)
        cleaned_data[['title', 'year']] = cleaned_data['title'].str.extract(r'^(.*) \((\d{4})\)$')
        cleaned_data['year'] = pd.to_numeric(cleaned_data['year'])  
        cleaned_data.drop(columns=['genres','userId', 'movieId', 'title', 'timestamp','zip-code'], inplace=True)
        self.merged = pd.get_dummies(cleaned_data, columns=['gender', 'occupation'], drop_first=True)
