import os
from settings import BASE_DIR, voc_dir
import torch
from models.model_wrapper import FasterRCNN, load_json, eval
import logging
from configs.faster_rcnn_config import opt
import copy
import gc
from data.voc.dataset import Dataset, TestDataset
from torch.utils.data import DataLoader
from configs.faster_rcnn_config import device, num_global_epoch

if device != "cpu":
    device = "cuda:3"

option = copy.deepcopy(opt)
option.voc_data_dir = voc_dir
test_dataloader = DataLoader(
    TestDataset(option, split='test'),
    batch_size=1,
    shuffle=False,
)
centralized_config = {
    "model_config": {
        "batch_size": 1
    },
    "data_path": opt.voc_data_dir,
    "local_epoch": 5
}
log_file = os.path.join(*[BASE_DIR, "logs", "baseline_faster_rcnn.txt"])
logging.basicConfig(filename=log_file, level=logging.INFO)

config_path = os.path.join(*[BASE_DIR, "configs"])
global_wrapper = FasterRCNN(task_config=centralized_config, device=device)

if __name__ == "__main__":
    epoch_map = []
    for epoch in range(1000):
        logging.info("==================epoch===================" + str(epoch + 1))
        total_loss = 0
        total_loss = global_wrapper.train_one_epoch()
        if device != "cpu":
            torch.cuda.empty_cache()
        logging.info("train: " + str(total_loss))
        if device != "cpu":
            torch.cuda.empty_cache()
        gc.collect()
        logging.info("===============global: =================")
        total_loss, result = eval(global_wrapper, test_dataloader, device=device, test_num=500)
        if device != "cpu":
            torch.cuda.empty_cache()
        map = result['map']
        ap = result['ap']
        epoch_map.append(map)
        logging.info("eval: " + str(total_loss) + " " + str(map))
