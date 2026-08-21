import numpy as np
import random as _random
from collections import defaultdict
from data.data import Data
from data.graph import Graph
import scipy.sparse as sp


class Interaction(Data, Graph):
    def __init__(self, conf, training, test):
        Graph.__init__(self)
        Data.__init__(self, conf, training, test)

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()
        # Validation set carved from the TRAINING interactions for model selection.
        self.valid_set = defaultdict(dict)
        self.valid_set_item = set()
        self.valid_data = []

        # Carve out the validation split BEFORE indices / graph are built so that
        # ui_adj / norm_adj / interaction_mat and all popularity / NMF statistics
        # are computed on the training portion only (no validation/test leakage).
        self.__split_validation(conf)
        self.__generate_set()
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()

    def __split_validation(self, conf):
        """Hold out a per-user fraction of the training interactions as a
        validation set. Uses a fixed split seed so the train/val partition is
        identical across model-init seeds. ratio<=0 disables (falls back to
        selecting on the test set, i.e. the original leaky behaviour)."""
        ratio = float(conf['valid.ratio']) if conf.contain('valid.ratio') else 0.1
        seed = int(conf['split.seed']) if conf.contain('split.seed') else 2024
        if ratio <= 0:
            return
        rng = _random.Random(seed)
        by_user = defaultdict(list)
        for rec in self.training_data:
            by_user[rec[0]].append(rec)
        train_keep, valid_hold = [], []
        for u, recs in by_user.items():
            n = len(recs)
            if n < 2:
                train_keep.extend(recs)
                continue
            n_val = max(1, int(round(ratio * n)))
            n_val = min(n_val, n - 1)  # always leave at least one interaction in train
            idx = list(range(n))
            rng.shuffle(idx)
            val_idx = set(idx[:n_val])
            for j, rec in enumerate(recs):
                (valid_hold if j in val_idx else train_keep).append(rec)
        self.training_data = train_keep
        self.valid_data = valid_hold

    def __generate_set(self):
        for user, item, rating in self.training_data:
            if user not in self.user:
                user_id = len(self.user)
                self.user[user] = user_id
                self.id2user[user_id] = user
            if item not in self.item:
                item_id = len(self.item)
                self.item[item] = item_id
                self.id2item[item_id] = item
            self.training_set_u[user][item] = 1
            self.training_set_i[item][user] = 1

        for user, item, rating in self.test_data:
            if user in self.user and item in self.item:
                self.test_set[user][item] = 1
                self.test_set_item.add(item)

        for user, item, rating in self.valid_data:
            if user in self.user and item in self.item:
                self.valid_set[user][item] = 1
                self.valid_set_item.add(item)

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        n_nodes = self.user_num + self.item_num
        user_np = np.array([self.user[pair[0]] for pair in self.training_data])
        item_np = np.array([self.item[pair[1]] for pair in self.training_data]) + self.user_num
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np)), shape=(n_nodes, n_nodes), dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        user_np_keep, item_np_keep = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_mat.shape[0])),
                                shape=(adj_mat.shape[0] + adj_mat.shape[1], adj_mat.shape[0] + adj_mat.shape[1]),
                                dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        row = np.array([self.user[pair[0]] for pair in self.training_data])
        col = np.array([self.item[pair[1]] for pair in self.training_data])
        entries = np.ones(len(row), dtype=np.float32)
        return sp.csr_matrix((entries, (row, col)), shape=(self.user_num, self.item_num), dtype=np.float32)

    def get_user_id(self, u):
        return self.user.get(u)

    def get_item_id(self, i):
        return self.item.get(i)

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def valid_size(self):
        return len(self.valid_set), len(self.valid_set_item), len(self.valid_data)

    def contain(self, u, i):
        return u in self.user and i in self.training_set_u[u]

    def contain_user(self, u):
        return u in self.user

    def contain_item(self, i):
        return i in self.item

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        k, v = self.user_rated(self.id2user[u])
        vec = np.zeros(self.item_num, dtype=np.float32)
        for item, rating in zip(k, v):
            vec[self.item[item]] = rating
        return vec

    def col(self, i):
        k, v = self.item_rated(self.id2item[i])
        vec = np.zeros(self.user_num, dtype=np.float32)
        for user, rating in zip(k, v):
            vec[self.user[user]] = rating
        return vec

    def matrix(self):
        m = np.zeros((self.user_num, self.item_num), dtype=np.float32)
        for u, u_id in self.user.items():
            vec = np.zeros(self.item_num, dtype=np.float32)
            k, v = self.user_rated(u)
            for item, rating in zip(k, v):
                vec[self.item[item]] = rating
            m[u_id] = vec
        return m
