import pandas as pd
from typing import List


class DataPrep:
    names_ratings: List[str] = ["UserID", "MovieID", "Rating", "Timestamp"]
    names_movies: List[str] = ["MovieID", "Title", "Genres"]

    def __init__(self, data_path: str):
        self.ratings: pd.DataFrame = pd.read_csv(data_path+ "/ratings.csv")
        self.movies: pd.DataFrame = pd.read_csv(data_path + "/movies.csv")

