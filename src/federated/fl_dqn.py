import copy
import math
from collections import deque, namedtuple
import random
from itertools import count

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from torch import nn, optim
from torch.utils.data import DataLoader

from federated.fl_base import FL
from federated.shortcuts import average_weights, flatten_weight
from models.base import ModelHandler
from utils.data import DatasetSplit
from env import train_loader, test_loader, user_groups, args, global_model

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 4
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 25  # 越大，衰减速度越慢
TARGET_UPDATE = 10


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


def get_reward(acc, base=64, target=0.95):
    return torch.tensor(np.power(base, acc - target) - 1).view(-1, 1)


class FLAgent(FL):
    def __init__(self, state_dim, args, train_loader, user_groups, test_loader, global_model, n_components=100):
        super().__init__(args, train_loader, user_groups, test_loader, global_model, "rl")
        self.name = "dqn"
        self.round = 1
        self.memory = ReplayMemory(32)
        self.policy_net = DQN(state_dim, self.num_users)
        self.target_net = DQN(state_dim, self.num_users)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.topk = max(int(args.frac * args.num_users), 1)
        self.pca = PCA(n_components=n_components)
        self.n_components = n_components
        self.state = np.array([flatten_weight(global_model).detach().numpy()])
        # 训练第一轮，使用PCA降维
        for idx in range(self.num_users):
            client = ModelHandler(
                model=copy.deepcopy(global_model),
                train_dl=DataLoader(
                    DatasetSplit(train_loader.dataset, user_groups[idx]),
                    batch_size=args.local_bs, shuffle=True), args=args)
            client.train()
            self.state = np.concatenate([
                self.state,
                np.array([flatten_weight(client.model).detach().numpy()])])
        self.state = self.pca.fit_transform(self.state)

    def get_current_state(self):
        return torch.tensor(self.state).reshape(-1, self.n_components * (self.num_users + 1))

    def get_users_idxs(self):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.round / EPS_DECAY)
        # print("以概率%.2f随机抽取客户端" % eps_threshold)
        state = self.get_current_state()
        if sample > eps_threshold:
            with torch.no_grad():
                users_idxs = self.policy_net(state).topk(1).indices.numpy().flatten()
                print("使用策略网络选取客户端：", users_idxs)
        else:
            users_idxs = np.random.choice(np.arange(self.num_users), 1)
            print("随机抽取客户端：", users_idxs)
        return users_idxs

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, torch.max(
            self.policy_net(non_final_next_states), 1)[1].unsqueeze(1)).detach().flatten()
        next_state_values = next_state_values.view(-1, 1)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.float())

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def run(self):
        round_acc = []
        state = self.get_current_state()
        init_episode_state = copy.deepcopy(self.state)
        init_episode_global_modeler = copy.deepcopy(self.global_model_handler)
        target_acc = 0.3
        for i in range(100):
            total_reward = 0
            for epoch in count():  # range(self.rounds):
                # 选择一批进行训练，并更新全局模型
                local_losses, local_weights, idxs_users = self.local_train()
                global_weights = average_weights(local_weights)
                self.global_model_handler.model.load_state_dict(global_weights)
                # 计算全局模型在测试集上的损失和准确率
                loss, acc = self.global_model_handler.validation()
                print("Episode {} Round {} acc: {:.2%} loss: {:.6f}".format(i, epoch + 1, acc, loss))
                reward = get_reward(acc)
                total_reward += reward.numpy().flatten()[0]
                round_acc.append(acc)

                # 更新状态
                local_weights = np.insert(local_weights, 0, global_weights)
                idxs_users = np.insert(idxs_users, 0, -1)
                for idx, weight in zip(idxs_users, local_weights):
                    w = torch.Tensor()
                    for k, v in weight.items():
                        w = torch.cat((w, torch.flatten(v)))
                    self.state[idx + 1] = self.pca.transform(w.reshape(1, -1))
                next_state = self.get_current_state()
                self.memory.push(state, torch.tensor(idxs_users[1]).view(-1, 1), next_state, reward)
                self.optimize_model()
                if epoch % TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                self.round += 1
                if acc >= target_acc:
                    self.round = 1
                    self.global_model_handler = copy.deepcopy(init_episode_global_modeler)
                    self.state = copy.deepcopy(init_episode_state)
                    print("Return: {}".format(total_reward))
                    break
        self.save_result(round_acc)


if __name__ == '__main__':
    n_components = 100
    state_dim = (args.num_users + 1) * n_components
    fl = FLAgent(state_dim, args, train_loader, user_groups, test_loader, global_model, n_components)
    fl.run()
