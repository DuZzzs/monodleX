import os
import torch
import torch.nn as nn


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state}


def save_checkpoint(state, filename):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename, map_location, logger=None):
    if os.path.isfile(filename):
        if logger:
            logger.info("==> Loading from checkpoint '{}'".format(filename))
        else:
            print("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location)
        # epoch = checkpoint.get('epoch', -1)
        checkpoint_dir, pth_file = os.path.split(filename)
        epoch = int(pth_file.split("_")[-1].split(".")[0])

        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if logger:
            logger.info("==> Done")
        else:
            print("==> Done")
    else:
        raise FileNotFoundError

    return epoch
