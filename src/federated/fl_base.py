import copy
import os

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from federated.shortcuts import average_weights
from models.base import ModelHandler
from settings import BASE_DIR
from utils.data import DatasetSplit


class FL:
    def __init__(self, args, train_loader, user_groups, test_loader, global_model, name="avg"):
        self.user_groups = user_groups
        self.frac = args.frac
        self.num_users = args.num_users
        self.name = name
        self.global_model_handler = ModelHandler(train_loader, test_loader, global_model, args)
        self.global_weights = self.global_model_handler.get_weight_dict()
        self.rounds = args.epochs
        self.train_loader = train_loader
        self.local_bs = args.local_bs
        self.args = args

    def get_users_idxs(self):
        m = max(int(self.frac * self.num_users), 1)
        idxs_users = np.random.choice(range(self.num_users), m, replace=False)
        return idxs_users

    def save_result(self, round_acc):
        plt.figure()
        plt.plot(range(len(round_acc)), round_acc)
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')
        plt.title('Test accuracy')
        save_dir = os.path.join(BASE_DIR, 'save')
        fig_name = 'fl_{}_{}_{}_{}.png'.format(self.name, self.args.dataset, self.args.model, self.rounds)
        plt.savefig(os.path.join(save_dir, fig_name))

    def local_train(self):
        local_weights, local_losses = [], []
        # 抽取一部分客户端
        idxs_users = self.get_users_idxs()

        for idx in idxs_users:
            client = ModelHandler(model=copy.deepcopy(self.global_model_handler.model),
                                  train_dl=DataLoader(
                                      DatasetSplit(self.train_loader.dataset, self.user_groups[idx]),
                                      batch_size=self.local_bs, shuffle=True),
                                  args=self.args)
            loss, acc = client.train()
            local_weights.append(copy.deepcopy(client.get_weight_dict()))
            local_losses.append(copy.deepcopy(loss))
        return local_losses, local_weights, idxs_users

    def run(self):
        round_acc = []
        for epoch in range(self.rounds):
            local_losses, local_weights, idxs_users = self.local_train()
            global_weights = average_weights(local_weights)
            self.global_model_handler.model.load_state_dict(global_weights)
            loss_avg = sum(local_losses) / len(local_losses)
            loss, acc = self.global_model_handler.validation()
            print("Round {} acc: {:.2%} loss: {:.6f}".format(epoch + 1, acc, loss))
            round_acc.append(acc)
        self.save_result(round_acc)
        return round_acc
