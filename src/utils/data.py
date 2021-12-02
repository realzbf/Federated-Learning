import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from settings import BASE_DIR


def split_and_shuffle_labels(y_data, seed):
    y_data = pd.DataFrame(y_data, columns=["labels"])
    # 获取索引
    y_data["i"] = np.arange(len(y_data))
    label_dict = dict()
    for i in range(10):
        # 取等于i的标签
        label_info = y_data[y_data["labels"] == i]
        np.random.seed(seed)
        # 随机排列
        label_info = np.random.permutation(label_info)
        label_dict.update({i: label_info[:, 1]})
    return label_dict


def get_user_groups_iid(targets, num_users, seed=2):
    labels_dict = split_and_shuffle_labels(targets, seed)
    user_groups_dict = dict()
    for user_id in range(num_users):
        user_groups_dict.update({user_id: np.array([])})

    for i in range(10):
        total_label_i = labels_dict[i].shape[0]
        amount_user_data = int(math.floor(total_label_i / num_users))
        for user_id in range(num_users):
            user_groups_dict[user_id] = np.concatenate((
                user_groups_dict[user_id], labels_dict[i][user_id * amount_user_data:(user_id + 1) * amount_user_data]
            )).astype(int)

    return user_groups_dict


def get_user_groups_non_iid(targets, num_users, n_class=2, num_samples=300, rate_unbalance=1):
    num_shards_train, num_imgs_train = int(len(targets) / num_samples), num_samples
    num_classes = 10
    assert (n_class * num_users <= num_shards_train)
    assert (n_class <= num_classes)
    idx_shard = [i for i in range(num_shards_train)]
    user_groups = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train * num_imgs_train)
    labels = np.array(targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                user_groups[i] = np.concatenate(
                    (user_groups[i], idxs[rand * num_imgs_train:(rand + 1) * num_imgs_train]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand * num_imgs_train:(rand + 1) * num_imgs_train]),
                                             axis=0)
            else:
                user_groups[i] = np.concatenate(
                    (user_groups[i], idxs[rand * num_imgs_train:int((rand + rate_unbalance) * num_imgs_train)]),
                    axis=0)
                user_labels = np.concatenate(
                    (user_labels, labels[rand * num_imgs_train:int((rand + rate_unbalance) * num_imgs_train)]), axis=0)
            unbalance_flag = 1

    return user_groups


def get_dataset(args, download=True):
    data_dir = os.path.join(BASE_DIR, 'data')
    if args.dataset == 'cifar':
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=download,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=download,
                                        transform=apply_transform)


    elif args.dataset == 'mnist' or args.dataset == 'fmnist':

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(data_dir, train=True, download=download,
                                           transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=download,
                                          transform=apply_transform)
        else:
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=download,
                                                  transform=apply_transform)

            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=download,
                                                 transform=apply_transform)

    else:
        raise Exception(f"无数据集：{args.dataset}")
    if not args.iid:
        user_groups = get_user_groups_non_iid(train_dataset.targets, args.num_users)
    else:
        user_groups = get_user_groups_iid(train_dataset.targets, args.num_users)
    return train_dataset, test_dataset, user_groups


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        data, target = self.dataset[self.idxs[item]]
        return data.clone().detach(), torch.tensor(target)


if __name__ == '__main__':
    from options import args_parser

    args = args_parser()
    args.dataset = 'fmnist'
    train, test, _ = get_dataset(args)
