from argparse import ArgumentParser, Namespace
from typing import Any, Tuple

from torch import __version__, cuda, backends
import sys

import torch_template.src.modules as models
from torch import optim as torch_optim

from torch.nn import Module as TorchModule


def get_model_from_config(model_config: dict, device):
    model = getattr(models, model_config.get("type"))
    model = model.from_config(model_config, device)
    return model


def get_optimizer_scheduler(config: dict, model: TorchModule) -> Tuple[torch_optim.Optimizer, Any]:
    # get optimizer, else None
    optimizer: str = config["optimizer"]
    optimizer: torch_optim.Optimizer = getattr(torch_optim, optimizer, None)(
        params=model.parameters(), lr=config["lr"], 
        weight_decay=config["weight_decay"])
    
    # get scheduler, else None
    scheduler: str = config.get("scheduler", None)
    # do we want a scheduler?
    if scheduler == "" or scheduler is None:
        scheduler = None
    else:
        scheduler = getattr(torch_optim.lr_scheduler, scheduler, None)(
            optimizer)

    return optimizer, scheduler


def command_line_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU instead of CPU")
    parser.add_argument(
        "--num_epochs", default=100, help="Number of epochs to train the model")
    parser.add_argument(
        "--task", default="train", help="Train a model or infer from the model")
    parser.add_argument(
        "--debug", action="store_true", 
        help="Generate more auxiliary files, output etc. for better understanding of training process")
    parser.add_argument(
        "--config", 
        default="config/config.toml", 
        help="Path to config from which the model, dataset etc. will be build")
    args = parser.parse_args()
    return args


def info_cuda():
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', __version__)
    print('__CUDA VERSION')
    from subprocess import call
    call(["/usr/local/cuda-11.3/bin/nvcc", "--version"])
    print('__CUDNN VERSION:', backends.cudnn.version())
    print('__Number CUDA Devices:', cuda.device_count())
    print('__Devices')
    call(["nvidia-smi", "--format=csv",
         "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', cuda.current_device())
    print('Available devices ', cuda.device_count())
    print('Current cuda device ', cuda.current_device())