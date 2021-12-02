import copy
import os
import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from federated.fl_base import FL
from settings import BASE_DIR
from shortcuts import average_weights, flatten_weight
from models.base import ModelHandler
from utils.data import DatasetSplit
from sklearn.mixture import GaussianMixture


class KCenterFL(FL):
    def __init__(self, args, train_loader, user_groups, test_loader, global_model):
        super().__init__(args, train_loader, user_groups, test_loader, global_model, "kcenter")
        # 聚类
        self.n_clusters = max(int(args.frac * args.num_users), 1)
        random_state = 2
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state)
        local_flatten_weights = []
        for idx in range(args.num_users):
            client = ModelHandler(
                model=copy.deepcopy(global_model),
                train_dl=DataLoader(
                    DatasetSplit(train_loader.dataset, user_groups[idx]),
                    batch_size=args.local_bs, shuffle=True), args=args)
            client.train()
            local_flatten_weights.append(flatten_weight(client.model).detach().numpy())
        kmeans.fit(local_flatten_weights)
        self.groups = dict()
        for i in range(self.n_clusters):
            self.groups[i] = []
        for idx, label in enumerate(kmeans.labels_):
            self.groups[label].append(idx)

    def get_users_idxs(self):
        idxs = []
        for j in range(self.n_clusters):
            random.seed(888)
            inx = random.choice(self.groups[j])
            idxs.append(inx)
        return idxs


if __name__ == '__main__':
    from env import global_model, test_loader, train_loader, args, user_groups

    fl = KCenterFL(args, train_loader, user_groups, test_loader, global_model)
    fl.run()
