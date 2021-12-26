import json
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.voc.dataset import Dataset, TestDataset
from utils.faster_rcnn import array_tool as at
from configs.faster_rcnn_config import opt
from models.faster_rcnn.faster_rcnn_vgg16 import FasterRCNNVGG16
from utils.faster_rcnn.trainer import FasterRCNNTrainer
from utils.faster_rcnn.eval_tool import eval_detection_voc


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


class FasterRCNN(object):
    """
    In fasterRCNN model, we only return the total loss, calculated from:
        rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss,
    and mAP@0.5
    """

    def __init__(self, task_config, cuda_device="cuda:0"):

        self.model_config = task_config['model_config']
        self.model_config['voc_data_dir'] = task_config['data_path']
        self.opt = opt
        self.device = cuda_device if opt.cuda else "cpu"
        self.opt._parse(self.model_config)

        # 数据集
        self.dataset = Dataset(self.opt)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.model_config['batch_size'],
                                     shuffle=True)
        self.testset = TestDataset(self.opt, split='train')
        self.test_dataloader = DataLoader(
            self.testset,
            batch_size=self.model_config['batch_size'],
            shuffle=False,
        )
        self.train_size = self.dataset.__len__()
        # self.valid_size = self.testset.__len__()

        # 模型
        self.faster_rcnn = FasterRCNNVGG16()
        self.trainer = FasterRCNNTrainer(
            self.faster_rcnn
        ).to(self.device)

        # 使用预训练模型
        if self.opt.load_path:
            self.trainer.load(self.opt.load_path)
            logging.info('load pretrained model from %s' % self.opt.load_path)
        self.best_map = 0
        self.lr_ = self.opt.lr

    def train_one_epoch(self):
        """
        Return:
            total_loss: the total loss during training
            accuracy: the mAP
        """
        self.trainer.reset_meters()
        for ii, (img, sizes, bbox_, label_, scale, gt_difficults_) in \
                tqdm(enumerate(self.dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.to(self.device).float(), bbox_.to(self.device), label_.to(self.device)
            self.trainer.train_step(img, bbox, label, scale)
            del img, bbox, label, scale, gt_difficults_, sizes
            if self.device != 'cpu':
                torch.cuda.empty_cache()

        return self.trainer.get_meter_data()['total_loss']

    def validate(self):
        """
        In the current version, the validate dataset hasn't been set,
        so we use the first 500 samples of testing set instead.
        """
        print("run validation")
        return self.evaluate(500)

    def evaluate(self, test_num=10000):
        """
        Return:
            total_loss: the average loss
            accuracy: the evaluation map
        """
        total_loss, eval_result = eval(self,
                                       self.test_dataloader, test_num)
        return total_loss, eval_result['map'], eval_result['ap']


def eval(self: FasterRCNN, dataloader, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    total_losses = list()
    with torch.no_grad():
        for ii, (imgs, sizes, gt_bboxes_, gt_labels_, scale, gt_difficults_) \
                in tqdm(enumerate(dataloader)):
            if self.device != "cpu":
                img = imgs.cuda().float()
                bbox = gt_bboxes_.cuda()
                label = gt_labels_.cuda()
            else:
                img = imgs.float()
                bbox = gt_bboxes_
                label = gt_labels_
            sizes = [sizes[0][0].item(), sizes[1][0].item()]
            pred_bboxes_, pred_labels_, pred_scores_ = \
                self.faster_rcnn.predict(imgs, [sizes])
            losses = self.trainer.forward(img, bbox, label, float(scale))
            total_losses.append(losses.total_loss.item())
            gt_bboxes += list(gt_bboxes_.numpy())
            gt_labels += list(gt_labels_.numpy())
            gt_difficults += list(gt_difficults_.numpy())
            pred_bboxes += pred_bboxes_
            pred_labels += pred_labels_
            pred_scores += pred_scores_
            if ii == test_num:
                break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    total_loss = sum(total_losses) / len(total_losses)
    return total_loss, result


if __name__ == "__main__":
    wrapper = FasterRCNN(
        task_config=load_json("/Users/zhengbangfeng/Documents/project/Federated-Learning/configs/faster_task1.json"))
    total_loss = wrapper.train_one_epoch()
    print("train: ", total_loss)
    total_loss, map, ap = wrapper.evaluate()
    print("eval: ", total_loss, map, ap)
