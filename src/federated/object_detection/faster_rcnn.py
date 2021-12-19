import os
from settings import BASE_DIR, voc_dir

from models.model_wrapper import FasterRCNN, load_json, eval
from federated.shortcuts import average_weights
import logging
from configs.faster_rcnn_config import opt
import copy
from data.voc.dataset import Dataset, TestDataset
from torch.utils.data import DataLoader

option = copy.deepcopy(opt)
option.voc_data_dir = voc_dir
test_dataloader = DataLoader(
    TestDataset(option, split='test'),
    batch_size=1,
    shuffle=False,
)

log_file = os.path.join(*[BASE_DIR, "logs", "avg_faster_rcnn.txt"])
logging.basicConfig(filename=log_file, level=logging.INFO)

street_5_tasks_path = os.path.join(*[BASE_DIR, "configs", "street_5"])
street_20_tasks_path = os.path.join(*[BASE_DIR, "configs", "street_20"])
weights = []
global_wrapper = None

for i in range(5):
    wrapper = FasterRCNN(
        task_config=load_json(os.path.join(street_20_tasks_path, "task" + str(i + 1) + ".json"))
    )
    total_loss = wrapper.train_one_epoch()
    logging.info("============client: ==============" + str(i + 1))
    logging.info("train: ", total_loss)
    total_loss, map, ap = wrapper.evaluate()
    logging.info("eval: ", total_loss, map, ap)
    weights.append(wrapper.faster_rcnn.state_dict())
    if global_wrapper is None:
        global_wrapper = wrapper

logging.info("===============global: =================")
weight = average_weights(weights)
global_wrapper.faster_rcnn.load_state_dict(weight)
total_loss, map, ap = global_wrapper.evaluate()
logging.info("eval: ", total_loss, map, ap)
