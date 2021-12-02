import copy
import random
from collections import Counter

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from federated.fl_base import FL
from federated.shortcuts import average_weights
from models.base import ModelHandler, CNNMnist
from settings import device
from utils.data import DatasetSplit
from env import get_model


class Discriminator(nn.Module):
    def __init__(self, length_feature=10, num_clients=100):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(length_feature, 200)
        self.fc2 = nn.Linear(200, 64)
        self.fc3 = nn.Linear(64, num_clients)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


class UADLoss(nn.Module):
    def __init__(self, num_clients):
        super(UADLoss, self).__init__()
        self.num_clients = num_clients

    def forward(self, t):
        eps = 1e-7
        return torch.div(-1, self.num_clients) * torch.sum(torch.log(t + eps))


class DiscriminatorLoss(nn.Module):
    def __init__(self, num_group_clients):
        super(DiscriminatorLoss, self).__init__()
        self.num_group_clients = num_group_clients

    def forward(self, d_self, d_other):
        eps = 1e-7
        loss = -torch.log(d_self + eps) - torch.div(1, self.num_group_clients - 1) * \
               torch.sum(torch.log(d_other + eps), dim=1).view(-1, 1)
        return torch.div(loss.sum(), d_self.shape[0])


def get_clients_p(clients, args):
    num_classes = args.num_classes
    total_count = Counter()
    num = len(clients)
    for i in range(num):
        total_count += clients[i].num_classes_dict
    result = []
    for i in range(num):
        class_frac = [0.0 for i in range(num_classes)]
        # label is type of tensor
        for label, count in clients[i].num_classes_dict.items():
            class_frac[label] = count / total_count[label]
        result.append(class_frac)
    return np.array(result)


def get_clients_batch_predict(X, clients, args):
    num = len(clients)
    batch_size = args.local_bs
    num_classes = args.num_classes
    batch_predict = np.zeros((batch_size, num, num_classes))
    for client_idx in range(num):
        client_batch_predict = clients[client_idx].model(X).detach().numpy()
        for idx, pred in enumerate(client_batch_predict):
            batch_predict[idx, client_idx, :] = pred
    return batch_predict


def get_mixed_predict(X, clients, args):
    clients_batch_predict = get_clients_batch_predict(X, clients, args)
    clients_p = get_clients_p(clients, args)
    result = []
    for clients_predict in clients_batch_predict:
        result.append(F.softmax(torch.tensor(np.sum(clients_predict * clients_p, axis=0)), dim=0).detach().tolist())
    return torch.tensor(result)


class Client:
    def __init__(self, train_dl, args, num_classes_dict, group_index, num_group_clients, global_index, model=None,
                 momentum=0.9, weight_decay=0.0002):
        self.global_index = global_index,
        self.group_index = group_index
        self.device = device
        if not model:
            self.model = get_model(args)
        else:
            self.model = model
        self.prior_model = copy.deepcopy(self.model)
        self.train_dl = train_dl
        self.poster_model = self.prior_model
        for param in self.poster_model.classifier.parameters():
            param.requires_grad = False
        batch = None
        for batch, label in train_dl:
            break
        shape = self.poster_model.feature_extractor(batch).shape
        self.discriminator = Discriminator(length_feature=shape[1] * shape[2] * shape[3],
                                           num_clients=args.num_users)
        self.learning_rate = args.lr
        self.momentum = momentum
        if args.optimizer == 'sgd':
            self.prior_optimizer = torch.optim.SGD(self.prior_model.parameters(), lr=self.learning_rate,
                                                   momentum=momentum, weight_decay=weight_decay)
        else:
            self.prior_optimizer = torch.optim.Adam(self.prior_model.parameters(), lr=self.learning_rate,
                                                    weight_decay=weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.num_classes_dict = num_classes_dict
        self.args = args
        self.uad_loss = UADLoss(num_clients=args.num_users).to(self.device)
        self.poster_optimizer = torch.optim.SGD(self.poster_model.parameters(), lr=self.learning_rate,
                                                momentum=momentum, weight_decay=weight_decay)
        self.discriminator_loss = DiscriminatorLoss(num_group_clients=num_group_clients).to(self.device)
        self.discriminator_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=self.learning_rate,
                                                       momentum=momentum, weight_decay=weight_decay)

    def prior_model_train(self):
        prior_model_handler = ModelHandler(train_dl=self.train_dl, test_dl=self.train_dl, model=self.prior_model,
                                           args=args, momentum=0.9, weight_decay=0.0002)
        return prior_model_handler.train()

    def ufo_model_train(self, group_clients):
        discriminator_batch_loss = []
        extractor_batch_loss = []
        num_groups = len(group_clients)
        self.discriminator.train()
        self.prior_model.train()
        self.poster_model.train()
        self.prior_model.to(self.device)
        self.poster_model.to(self.device)
        self.discriminator.to(self.device)
        for batch_idx, (data, target) in enumerate(self.train_dl):
            data, target = data.to(self.device), target.to(self.device)
            y_hat = self.prior_model(data).to(self.device)
            cls = self.criterion(y_hat, target)  # 交叉损失

            # 计算UAD损失
            d_j_batch_list = []
            for j in range(num_groups):
                if j == self.group_index:
                    continue
                assert hasattr(group_clients[j], "prior_model")
                f_j_batch = group_clients[j].prior_model.feature_extractor(data).detach().to(self.device)
                d_j_batch = self.discriminator(f_j_batch)[:, group_clients[j].global_index].to(self.device)
                d_j_batch_list.append(d_j_batch.view(1, -1))
            d_j_batch = torch.cat(d_j_batch_list).T.detach().to(self.device)
            f_self = self.poster_model.feature_extractor(data).to(self.device)
            d_self = self.discriminator(f_self).detach().to(self.device)
            extractor_loss = cls + self.uad_loss(d_self)

            # 优化特征提取器
            self.poster_optimizer.zero_grad()
            extractor_loss.backward()
            self.poster_optimizer.step()

            # 优化判别器
            f_self = self.poster_model.feature_extractor(data).detach().to(self.device)
            d_self = self.discriminator(f_self)[:, self.global_index].view(-1, 1).to(self.device)
            extractor_batch_loss.append(extractor_loss.item())
            discriminator_loss = self.discriminator_loss(d_self, d_j_batch)
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_batch_loss.append(discriminator_loss.item())
            self.discriminator_optimizer.step()

        return sum(extractor_batch_loss) / len(extractor_batch_loss), sum(discriminator_batch_loss) / len(
            discriminator_batch_loss)

    # def poster_model_train(self, group_clients):
    #     self.poster_model.train()
    #     batch_loss = []
    #     for batch_idx, (data, target) in enumerate(self.train_dl):
    #         data = data.to(self.device)
    #         target = get_mixed_predict(data, group_clients, args)
    #         output = self.poster_model(data)
    #         loss = F.kl_div(torch.log(output), target, reduction='sum')
    #         loss.backward()
    #         # self.optimizer.step()
    #         batch_loss.append(loss.item())
    #     loss_avg = sum(batch_loss) / len(batch_loss)
    #     return loss_avg


def plot(results, labels):
    plt.figure()
    plt.ylabel("Test Accuracy")
    plt.xlabel("Round")
    num = results.shape[1]
    for i in range(num):
        y = np.around(results[:, i].astype('float'), 2)
        plt.plot(y, label=labels[i])
    plt.legend()
    plt.show()


from env import test_loader, train_loader, args, user_groups

"""分组"""
num_clients = args.num_users
num_group_clients = int(args.frac * num_clients)
num_groups = num_clients // num_group_clients


def get_group(idxs):

    return [
        Client(train_dl=DataLoader(
            DatasetSplit(train_loader.dataset, user_groups[idx]),
            batch_size=args.local_bs, shuffle=True),
            args=args,
            num_classes_dict=Counter(torch.tensor(train_loader.dataset.targets)[user_groups[idx]].detach().numpy()),
            group_index=0,  # 后续需动态修改
            num_group_clients=num_group_clients,
            global_index=idx
        )
        for idx in idxs
    ]


global_model = get_model(args)
ufo_global_model_handler = ModelHandler(train_dl=train_loader, test_dl=test_loader,
                                        model=copy.deepcopy(global_model),
                                        args=args)
avg_global_model_handler = ModelHandler(train_dl=train_loader, test_dl=test_loader,
                                        model=copy.deepcopy(global_model),
                                        args=args)
# half_group = random.sample(clients, num_group_clients // 2)
# for i in range(num_group_clients // 2):
#     half_group[i].group_index = i

for epoch in range(100):
    indices = random.sample(range(num_clients), num_group_clients)
    avg_group = get_group(indices)
    avg_weights = []
    loss, acc = 0, 0

    # 联邦平均
    for client in avg_group:
        # 下载模型
        client.prior_model.load_state_dict(avg_global_model_handler.model.state_dict())
        for u in range(10):
            loss, acc = client.prior_model_train()
        avg_weights.append(client.prior_model.state_dict())

    # UFO
    ufo_weights = []
    # 定义组索引
    ufo_group = get_group(indices)
    for i in range(num_group_clients):
        ufo_group[i].group_index = i
    for client in ufo_group:
        # 下载模型
        client.prior_model.load_state_dict(ufo_global_model_handler.model.state_dict())
    # 本地训练十轮
    for u in range(10):
        for idx,client in enumerate(ufo_group):
            # 分发自己的模型
            shared_group = copy.deepcopy(ufo_group)
            extractor_loss, discriminator_loss = client.ufo_model_train(shared_group)
            print("epoch{} :, client {} extractor_loss = {:.6f} discriminator_loss = {:.6f}".format(
                epoch, u, extractor_loss, discriminator_loss))
    for client in ufo_group:
        ufo_weights.append(client.poster_model.state_dict())

    avg_global_weights = average_weights(avg_weights)
    avg_global_model_handler.model.load_state_dict(avg_global_weights)
    print("epoch{} FedAVG global acc={:.3f}".format(epoch, avg_global_model_handler.validation()[1]))
    ufo_global_weights = average_weights(ufo_weights)
    ufo_global_model_handler.model.load_state_dict(ufo_global_weights)
    print("epoch{} FedUFO global acc={:.3f}".format(epoch, ufo_global_model_handler.validation()[1]))
