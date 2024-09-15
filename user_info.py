import pandas as pd


class UserInfo:

    def __init__(self):
        pass

    @classmethod
    def load_user_info(cls, name="TMBD"):

        ratings = pd.read_csv(f"./data/{name}.csv")
        ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]

        user_info = ratings[["user_id"]].drop_duplicates()
        user_info.reset_index(inplace=True, drop=True)
        return user_info
