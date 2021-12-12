import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # 数据集配置
    parser.add_argument('--dataset', type=str, default='fmnist', help="name \
                            of dataset")
    parser.add_argument('--num_samples', type=int, default=300,
                        help="number of samples")
    parser.add_argument('--num_classes', type=int, default=2,
                        help="number of class")

    # 客户端配置
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")

    # 模型配置
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0002,
                        help='SGD weight_decay (default: 0.0002)')

    # 全局配置
    parser.add_argument('--rounds', type=int, default=300,
                        help="number of rounds of training")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument("--optimizer", type=str, default="sgd",
                        help="optimizer")

    parser.add_argument('--baseline_epochs', type=int, default=300,
                        help="number of epochs of baseline")

    args = parser.parse_args()
    return args
