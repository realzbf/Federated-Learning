from federated.fl_base import FL
from env import train_loader, test_loader, user_groups, args, global_model

if __name__ == '__main__':
    fl = FL(args, train_loader, user_groups, test_loader, global_model)
    fl.run()
