import json
import os.path
from settings import BASE_DIR
import copy

local_epoch = 5
batch_size = 1

template = {
    "model_config": {
        "batch_size": batch_size
    },
    "data_path": "/Users/zhengbangfeng/Documents/datasets/Street_Dataset/street_20/9/",
    "local_epoch": local_epoch
}

if __name__ == "__main__":
    base_data_path = "/Users/zhengbangfeng/Documents/datasets/Street_Dataset/"
    street_5_data_path = os.path.join(base_data_path, "street_5")
    street_20_data_path = os.path.join(base_data_path, "street_20")

    street_5_tasks_path = os.path.join(*[BASE_DIR, "configs", "street_5"])
    if not os.path.exists(street_5_tasks_path):
        os.mkdir(street_5_tasks_path)

    street_20_tasks_path = os.path.join(*[BASE_DIR, "configs", "street_20"])
    if not os.path.exists(street_20_tasks_path):
        os.mkdir(street_20_tasks_path)

    for i in range(5):
        t = copy.deepcopy(template)
        t["data_path"] = os.path.join(street_5_data_path, str(i + 1))
        json.dump(t, open(os.path.join(street_5_tasks_path, "task" + str(i + 1) + ".json"), "w"), indent=4)

    for i in range(20):
        t = copy.deepcopy(template)
        t["data_path"] = os.path.join(street_20_data_path, str(i + 1))
        json.dump(t, open(os.path.join(street_20_tasks_path, "task" + str(i + 1) + ".json"), "w"), indent=4)

