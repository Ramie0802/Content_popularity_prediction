import json
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm


def load_glove(
    words=None,
    glove_path="./glove.6B/glove.6B.50d.txt",
    save_path="./data/glove.json",
):

    if "glove.json" in os.listdir("./data/"):
        with open(save_path, "r") as f:
            glove = json.load(f)
            f.close()
    else:
        glove = {}
        with open(glove_path, "r", encoding="utf8") as f:
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                if words is None or word in words:
                    vector = np.asarray(values[1:], "float32")
                    glove[word] = vector.tolist()
            f.close()

        with open(save_path, "w") as f:
            json.dump(glove, f)
            f.close()

    return glove


def load_dataset(path):

    if "dataset.pkl" in os.listdir("./data/"):
        with open("./data/dataset.pkl", "rb") as f:
            dataset = pickle.load(f)
            f.close()
    else:
        with open(os.path.join(path, "u.genre"), "r") as f:
            data = f.readlines()
            genres_list = [
                (
                    d.strip().split("|")[0].lower().replace("'", "")
                    if d.find("oir") == -1
                    else "noir"
                )
                for d in data
            ][
                1:
            ]  # skip the "unknown" genre

        ratings = pd.read_csv(
            os.path.join(path, "u.data"),
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
        )

        ratings = ratings.dropna()
        ratings = ratings.drop_duplicates(subset=["user_id", "item_id"])
        ratings = ratings.sort_values(by=["timestamp"])

        movies = pd.read_csv(
            os.path.join(path, "u.item"),
            sep="|",
            encoding="latin-1",
            names=[
                "item_id",
                "title",
                "release_date",
                "video_release_date",
                "IMDb_URL",
                "unknown",
                *genres_list,
            ],
        )
        users = pd.read_csv(
            os.path.join(path, "u.user"),
            sep="|",
            names=["user_id", "age", "gender", "occupation", "zip_code"],
        )

        dataset = movies.drop(
            ["release_date", "title", "video_release_date", "IMDb_URL"],
            axis=1,
            inplace=False,
        )

        users = users.drop(["zip_code"], axis=1, inplace=False)

        # drop row if unknown is 1
        dataset = dataset[dataset["unknown"] != 1]
        dataset = dataset.drop(["unknown"], axis=1, inplace=False)

        # drop duplicates movies
        dataset = dataset.drop_duplicates(subset="item_id")
        # dataset = dataset.dropna()

        # remove ratings with unknown movies
        ratings = ratings[ratings["item_id"].isin(dataset["item_id"])]
        ratings = ratings[ratings["user_id"].isin(users["user_id"])]

        # reduce movie id by min to have a content id starting from 0
        min_id = min(dataset["item_id"])
        dataset["item_id"] = dataset["item_id"] - min_id
        ratings["item_id"] = ratings["item_id"] - min_id

        # handling missing item to ensure that the item_id is continuous
        substraction = 0

        for i in range(max(dataset["item_id"]) + 1):
            if i not in dataset["item_id"].values:
                substraction += 1

            dataset.loc[dataset["item_id"] == i, "item_id"] -= substraction
            ratings.loc[ratings["item_id"] == i, "item_id"] -= substraction

        # create sparse and semantic vectors
        sparse_vecs = {}
        semantic_vecs = {}

        glove = load_glove(words=genres_list)

        for index, row in dataset.iterrows():
            # extract one-hot encoded genres
            sparse_vec = row[1:].to_numpy()  # (18,)

            semantic_vec = []
            for genre in genres_list:
                if row[genre] == 1:
                    semantic_vec.append(glove[genre])

            # mean aggregation
            semantic_vec = np.mean(semantic_vec, axis=0)

            sparse_vecs[row["item_id"]] = sparse_vec
            semantic_vecs[row["item_id"]] = semantic_vec

        dataset.drop(genres_list, axis=1, inplace=True)

        cosine_sparse = {}
        cosine_semantic = {}

        item_ids = dataset["item_id"].to_numpy()
        item_ids.sort()

        # print(max(item_ids), min(item_ids), len(item_ids))

        # for i in range(max(item_ids) + 1):
        #     if i not in item_ids:
        #         print("missing ", i)

        sparse_matrix = np.array([sparse_vecs[item_id] for item_id in item_ids])
        semantic_matrix = np.array([semantic_vecs[item_id] for item_id in item_ids])

        # Normalize the vectors
        sparse_norms = np.linalg.norm(sparse_matrix, axis=1)
        semantic_norms = np.linalg.norm(semantic_matrix, axis=1)

        # Compute cosine similarities
        cosine_sparse_matrix = np.dot(sparse_matrix, sparse_matrix.T) / np.outer(
            sparse_norms, sparse_norms
        )
        cosine_semantic_matrix = np.dot(semantic_matrix, semantic_matrix.T) / np.outer(
            semantic_norms, semantic_norms
        )

        dataset = {
            "sparse_vec": sparse_vecs,
            "semantic_vec": semantic_vecs,
            "cosine_sparse": cosine_sparse_matrix,
            "cosine_semantic": cosine_semantic_matrix,
            "ratings": ratings,
            "users": users,
        }
        with open("./data/dataset.pkl", "wb") as f:
            pickle.dump(
                dataset,
                f,
            )
            f.close()

    return dataset
