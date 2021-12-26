import os
from settings import BASE_DIR, voc_dir
import torch
from models.model_wrapper import FasterRCNN, load_json, eval
from federated.shortcuts import average_weights
import logging
from configs.faster_rcnn_config import opt
import copy
import gc
from data.voc.dataset import Dataset, TestDataset
from torch.utils.data import DataLoader
from configs.faster_rcnn_config import device

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
global_wrapper = FasterRCNN(task_config=load_json(os.path.join(street_5_tasks_path, "task1.json")), device=device)

num_epochs = 1000
epoch_map = []
for epoch in range(num_epochs):
    logging.info("==================epoch===================" + str(epoch + 1))
    for i in range(5):
        if device == "cpu":
            wrapper = FasterRCNN(
                task_config=load_json(os.path.join(street_5_tasks_path, "task" + str(i + 1) + ".json")),
                device="cpu"  # "cuda:" + str(i % 4)
            )
        else:
            wrapper = FasterRCNN(
                task_config=load_json(os.path.join(street_5_tasks_path, "task" + str(i + 1) + ".json")),
                device="cuda:" + str(i % 4)
            )
        wrapper.faster_rcnn.load_state_dict(wrapper.faster_rcnn.state_dict())
        for j in range(5):
            total_loss = wrapper.train_one_epoch()
        if option.cuda:
            torch.cuda.empty_cache()
        logging.info("============client: ==============" + str(i + 1))
        logging.info("train: " + str(total_loss))
        total_loss, result = eval(wrapper, test_dataloader, test_num=500)
        if option.cuda:
            torch.cuda.empty_cache()
        map = result['map']
        ap = result['ap']
        logging.info("eval: " + str(total_loss) + " " + str(map))
        weights.append(wrapper.faster_rcnn.state_dict())
        del wrapper
        if option.cuda:
            torch.cuda.empty_cache()
        gc.collect()

    logging.info("===============global: =================")
    weight = average_weights(weights)
    global_wrapper.faster_rcnn.load_state_dict(weight)
    total_loss, result = eval(global_wrapper, test_dataloader, test_num=500)
    if option.cuda:
        torch.cuda.empty_cache()
    map = result['map']
    ap = result['ap']
    epoch_map.append(map)
    logging.info("eval: " + str(total_loss) + " " + str(map))
