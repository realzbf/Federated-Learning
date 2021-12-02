import copy

import torch
from torch import nn
import torch.nn.functional as F

from settings import device


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.feature_extractor(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class CNNFashionMnist(nn.Module):
    def __init__(self):
        super(CNNFashionMnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.feature_extractor = nn.Sequential(
            self.layer1,
            self.layer2
        )
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar10(nn.Module):
    def __init__(self):
        super(CNNCifar10, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(6, 16, 5),
            nn.Dropout2d(),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name='VGG11'):
        super(VGG, self).__init__()
        self.feature_extractor = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class ModelHandler:
    def __init__(self, train_dl=None, test_dl=None, model=None, args=None, momentum=0.5, weight_decay=1e-4):
        self.learning_rate = args.lr
        self.momentum = momentum
        self.device = device
        self.model = model.to(self.device)
        if args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=momentum)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.train_dl = train_dl
        self.test_dl = test_dl

    def validation(self):
        self.model.eval()
        batch_loss = []
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_dl):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                batch_loss.append(self.criterion(output, target).item())
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(target.view_as(prediction)).sum().item()

        return sum(batch_loss) / len(batch_loss), correct / len(self.test_dl.dataset)

    def train(self, epoch=None, print_log=False):
        self.model.train()
        batch_loss = []
        correct = 0
        for batch_idx, (data, target) in enumerate(self.train_dl):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if print_log and epoch is not None and batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx * len(data), len(self.train_dl.dataset),
                    100. * batch_idx / len(self.train_dl), loss.item()))
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss) / len(batch_loss)
        acc = correct / len(self.train_dl.dataset)
        return loss_avg, acc

    def get_weight_dict(self):
        return self.model.state_dict()
