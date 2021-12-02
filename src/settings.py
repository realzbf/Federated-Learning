import os

import torch.cuda

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == '__main__':
    print(BASE_DIR)
