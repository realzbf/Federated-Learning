import os

from matplotlib import pyplot as plt
from tqdm import tqdm

from models.base import ModelHandler
from settings import BASE_DIR

if __name__ == '__main__':
    from env import global_model, test_loader, train_loader, args

    model_handler = ModelHandler(train_loader, test_loader, global_model, args)
    epoch_train_acc = []
    epoch_test_acc = []
    epoch_train_loss = []
    for epoch in tqdm(range(args.epochs)):
        train_avg_loss, train_acc = model_handler.train(epoch, print_log=True)
        test_avg_loss, test_acc = model_handler.validation()
        epoch_train_loss.append(train_avg_loss)
        epoch_test_acc.append(test_acc)
        epoch_train_acc.append(train_acc)
        print("Train accuracy: {}\nTest accuracy: {}".format(train_acc, test_acc))

    plt.figure()
    plt.plot(range(len(epoch_test_acc)), epoch_test_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test accuracy')
    save_dir = os.path.join(BASE_DIR, 'save')
    fig_name = 'nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs)
    plt.savefig(os.path.join(save_dir, fig_name))
