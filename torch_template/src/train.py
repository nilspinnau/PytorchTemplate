from argparse import Namespace
from typing import Tuple
import sys

import torch_template.src.utils as utils
import torch_template.src.data.dataset as template_datasets

import torch
import time

from toml import load as load_toml


def train(args: Namespace, device: torch.DeviceObjType):

    # get all relevant configs
    config = load_toml(args.config)

    model_config = config.get("model")
    data_config = config.get("dataset")
    general_config = config.get("general")

    print("model_config: ", model_config)
    print("data_config: ", data_config)
    print("general_config: ", general_config)

    # get model
    model = utils.get_model_from_config(model_config, device)

    # get dataset
    dataset = getattr(template_datasets, str(data_config["type"]))
    dataset = dataset.from_config(data_config)
    
    # get optimizer and scheduler
    optimizer, scheduler = utils.get_optimizer_scheduler(general_config, model)
    
    # get loss function, else None
    loss_fn: str = general_config["loss_fn"]
    loss_fn = getattr(torch.nn, loss_fn, None)

    # if optimizer or loss function are None exit with failure
    if optimizer is None or loss_fn is None:
        print("Optimizer or loss function are None, thus we cannot train a neural network. Exit")
        sys.exit(1)

    num_epochs = general_config.get("num_epochs", args.num_epochs)

    # finally train the model
    start_time = time.time()
    train_loss, val_loss, test_loss = train_loop(
        dataset=dataset,
        model=model, optimizer=optimizer, loss_fn=loss_fn, scheduler=scheduler,
        num_epochs=num_epochs,
        device=device)
    elapsed_time = time.time() - start_time 


    print(
        "The model training finished with:\n"
        "\ttraining loss: {}\n"
        "\tvalidation loss: {}\n"
        "\ttest loss: {}\n"
        "\telapsed time: {}", train_loss, val_loss, test_loss, elapsed_time)
    return 0


def train_loop(
    model: torch.nn.Module, 
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    scheduler,
    num_epochs: int,
    device: torch.DeviceObjType) -> Tuple[float, float, float]:


    train_dataloader, val_dataloader, test_dataloader = dataset.get_dataloaders()

    train_loss, val_loss, test_loss = 0.0, 0.0, 0.0


    start_time = time.time()

    for _, epoch in enumerate(num_epochs):

        epoch_time = time.time()

        # training
        model.train()
        for train_data in train_dataloader:
            optimizer.zero_grad()

            loss = predict_loss_for_model(train_data, model, loss_fn, optimizer)
            
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            print("Epoch: {}, Train Loss: {}, Memory usage: {}", epoch, train_loss, 0)
        # validation
        model.eval()
        for val_data in val_dataloader:
            with torch.no_grad():
                loss = predict_loss_for_model(val_data, model, loss_fn, optimizer)
                val_loss = loss.item()
                if scheduler is not None:
                    scheduler.step(val_loss)

        print(
            "Epoch: {} finished, "
            "Memory consumption: {}, " 
            "Time elapsed for epoch: {}, " 
            "Total time elapsed: {}", 
            epoch, 0, time.time() - epoch_time, time.time() - start_time)

    # testing
    model.eval()
    for test_data in test_dataloader:
        with torch.no_grad():
            loss = predict_loss_for_model(test_data, model, loss_fn, optimizer)
            test_loss = loss.item()

    return train_loss, val_loss, test_loss



def predict_loss_for_model(
    data,
    model,
    loss_fn) -> float:

    input_data, target = data 
    prediction = model(input_data)
    loss = loss_fn(prediction, target)
    return loss