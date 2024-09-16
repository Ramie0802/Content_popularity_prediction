import time
import os
import pickle
import shutil
import numpy as np

import pandas as pd

from itertools import chain
from collections import Counter


class LogTime:

    def __init__(self, print_step=20000, words=""):
        self.process_count = 0
        self.PRINT_STEP = print_step()

        # record the calculate time has spent
        self.start_time = time.time()
        self.words = words
        self.total_time = 0.0

    def count_time(self):
        if self.process_count % self.PRINT_STEP == 0:
            curr_time = time.time()
            print(
                self.words
                + " steps(%d). %.2f seconds have spent.."
                % (self.process_count, curr_time - self.start_time)
            )
            self.process_count += 1

    def finish(self):
        print("total %s step number is %d" % (self.words, self.get_curr_step()))
        print("total %.2f seconds have spent\n" % self.get_total_time())

    def get_curr_step(self):
        return self.process_count

    def get_total_time(self):
        return time.time() - self.start_time


class ModelManager:

    path_name = ""

    def __init__(self, folder_name=None):

        self.folder_name = folder_name
        if not self.path_name:
            self.path_name = "model/" + folder_name + "/"

    def save_model(self, model, save_name: str):

        if "pkl" not in save_name:
            save_name += ".pkl"
        if not os.path.exists("model/"):
            os.mkdir("model/")
        if not os.path.exists(self.path_name):
            os.mkdir(self.path_name)
        if os.path.exists(self.path_name + "%s" % save_name):
            os.remove(self.path_name + "%s" % save_name)
        pickle.dump(model, open(self.path_name + "%s" % save_name, "wb"))

    def load_model(self, model_name: str):

        if "pkl" not in model_name:
            model_name == ".pkl"

        if not os.path.exists(self.path_name + "%s" % model_name):
            raise OSError("There is no model named %s in model/ dir" % model_name)
        return pickle.load(open(self.path_name + "%s" % model_name, "rb"))

    def clean_workspace(self, clean=False):

        if clean and os.path.exists(self.path_name):
            shutil.rmtree(self.path_name)

    def delete_file(self, file_name):
        if "pkl" not in file_name:
            file_name += ".pkl"
        my_file = self.path_name + "-%s" % file_name
        if os.path.exists(my_file):
            os.remove(my_file)
        else:
            print("no such file:%s" % my_file)


class UserInfoManager:

    path_name = ""

    @classmethod
    def __init__(cls, user_info_name=None):

        if not cls.path_name:
            cls.path_name = "user/" + user_info_name

    def save_user_info(self, user_info, save_name: str):

        if "csv" not in save_name:
            save_name += ".csv"
        if not os.path.exists("user"):
            os.mkdir("user")
        pickle.dump(user_info, open(self.path_name + "-%s" % save_name, "wb"))

    def load_user_info(self, user_info_name: str):

        if "csv" not in user_info_name:
            user_info_name += ".csv"
        if not os.path.exists(self.path_name + "-%s" % user_info_name):
            raise OSError(
                "There is no user info named %s in model/ dir" % user_info_name
            )
        return pickle.load(open(self.path_name + "-%s" % user_info_name, "rb"))

    @staticmethod
    def clean_workspace(clean=False):
        if clean and os.path.exists("user"):
            shutil.rmtree("user")


def exp_details(args):
    print("\nExperimental details:")


def get_top_items(df, top_persent):

    top_25_percentile_value = df["popularity"].quantile(1 - top_persent)
    top_25_percent_rows = df[df["popularity"] >= top_25_percentile_value]
    # top_25_percent_movies = top_25_percent_rows['movie_id'].to_list()
    top_25_percent_movies = top_25_percent_rows[["movie_id", "popularity"]]

    return top_25_percent_movies


def concate3branch(df1, df2, df3):

    df1_renamed = df1[["movie_id", "popularity"]].rename(
        columns={"popularity": "popularity1"}
    )
    df2_renamed = df2[["movie_id", "popularity"]].rename(
        columns={"popularity": "popularity2"}
    )
    df3_renamed = df3[["movie_id", "popularity"]].rename(
        columns={"popularity": "popularity3"}
    )

    concatenated_df = pd.concat(
        [
            df1_renamed.set_index("movie_id"),
            df2_renamed.set_index("movie_id"),
            df3_renamed.set_index("movie_id"),
        ],
        axis=1,
    )

    return concatenated_df
