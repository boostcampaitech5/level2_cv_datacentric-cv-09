import random
import torch
import numpy as np

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_gpu():
    GB = 1024.*1024.0*1024.0
    return round(torch.cuda.max_memory_allocated() / GB, 1)