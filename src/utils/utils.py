import random
import string

import torch


def safe_div(x,y):
    if y == 0:
        return None
    return x / y

def random_string(length):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def cuda_else_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if torch.backends.mps.is_available():
        return tensor.cuda()
    if torch.backends.cuda.is_built():
        return tensor.cuda()
    return tensor.cpu()
