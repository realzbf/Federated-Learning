import torch
from torch.utils.data import DataLoader

from models.base import CNNMnist, CNNFashionMnist, CNNCifar10, MLP, VGG
from options import args_parser
from utils.data import get_dataset


def get_model(args):
    model = None
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            model = CNNMnist()
        elif args.dataset == 'fmnist':
            model = CNNFashionMnist()
        elif args.dataset == 'cifar':
            model = VGG()
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        model = MLP(dim_in=len_in, dim_hidden=64,
                    dim_out=args.num_classes)
    else:
        raise Exception('无该模型')
    return model.to(device)


args = args_parser()
train_dataset, test_dataset, user_groups = get_dataset(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
global_model = get_model(args)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
