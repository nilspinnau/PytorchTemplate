from argparse import Namespace

from random import seed as random_seed
from numpy.random import seed as numpy_seed

import torch

# set all seeds
numpy_seed(0)
random_seed(0)
torch.manual_seed(0)

from torch_template.src import infer
from torch_template.src import train
from torch_template.src.utils import command_line_args, info_cuda


def main(args: Namespace):
    # check the task 

    # we will pass the device to all relevant classes, so we create tensors on the device directly
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    if device == "cuda": 
        info_cuda()

    if args.task == "infer":
        infer.infer(args, device)
    elif args.task == "train":
        train.train(args, device)
    else:
        print("Unknown task. Exiting")
        return 1
    return 0

if __name__ == "__main__":
    args = command_line_args()
    main(args)