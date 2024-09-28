import numpy as np

from dataset import load_dataset


class ContentLibrary:
    def __init__(self, path):
        self.dataset = load_dataset(path)
        self.ratings = self.dataset["ratings"]
        self.users = self.dataset["users"]
        self.semantic_vecs = self.dataset["semantic_vec"]
        self.sparse_vecs = self.dataset["sparse_vec"]
        self.semantic_cosine = self.dataset["cosine_semantic"]
        self.sparse_cosine = self.dataset["cosine_sparse"]

        self.total_items = list(set(self.ratings["item_id"]))
        self.total_users = list(set(self.ratings["user_id"]))

        self.user_sorted = self.ratings["user_id"].value_counts()

        self.used_user_ids = []

        self.max_item_id = max(self.total_items)

    def load_ratings(self, user_id):
        ratings = self.ratings[self.ratings["user_id"] == user_id].copy()
        ratings.loc[:, "semantic_vec"] = ratings["item_id"].apply(
            lambda x: self.semantic_vecs[int(x)]
        )
        ratings.loc[:, "sparse_vec"] = ratings["item_id"].apply(
            lambda x: self.sparse_vecs[int(x)]
        )

        ratings = ratings.sort_values(by="timestamp", ascending=False)

        return {
            "contents": ratings["item_id"].values,
            "ratings": ratings["rating"].values,
            "semantic_vecs": ratings["semantic_vec"].values,
            "sparse_vecs": ratings["sparse_vec"].values,
            "max": self.max_item_id,
        }

    def load_user_info(self, user_id):
        return self.ratings[self.ratings["user_id"] == user_id]

    def get_user(self):
        user_id = np.random.choice(self.total_users)
        while user_id in self.used_user_ids:
            user_id = np.random.choice(self.total_users)
        self.used_user_ids.append(user_id)
        return user_id

    def return_user(self, user_id):
        self.used_user_ids.remove(user_id)
