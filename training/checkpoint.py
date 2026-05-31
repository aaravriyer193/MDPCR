# MDPCR — training/checkpoint.py
import torch
import config


def save_checkpoint(model, optimiser, epoch: int, path: str) -> None:
    torch.save({
        "epoch"       : epoch,
        "model_state" : model.state_dict(),
        "optim_state" : optimiser.state_dict(),
    }, path)


def load_checkpoint(model, optimiser, path: str) -> int:
    checkpoint = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    optimiser.load_state_dict(checkpoint["optim_state"])
    return checkpoint["epoch"]
