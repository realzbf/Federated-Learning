from env import train_loader, test_loader, user_groups, args, global_model
from federated.fl_base import FL
from federated.fl_kcenter import KCenterFL
import copy
if __name__ == '__main__':
    avg_fl = FL(args, train_loader, user_groups, test_loader, copy.deepcopy(global_model))
    avg_fed_acc_round = avg_fl.run()
    kcenter_fl = KCenterFL(args, train_loader, user_groups, test_loader, copy.deepcopy(global_model))
    kcenter_fed_acc_round = kcenter_fl.run()
