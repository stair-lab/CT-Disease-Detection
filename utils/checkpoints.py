import os
import shutil
import logging
import torch

# Functions in this file are inspired by the following:
# https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py

logger = logging.getLogger(__name__)


def save_checkpoint(state, model_state, isbest, checkpoint):
    """
    Save training and model state to a checkpoint directory.
    """
    filepath = os.path.join(checkpoint, 'last.pth')
    model_filepath = os.path.join(checkpoint, 'model_last.pth')
    if not os.path.exists(checkpoint):
        logger.info("Checkpoint directory does not exist. Creating %s", checkpoint)
        os.makedirs(checkpoint)

    torch.save(state, filepath)
    torch.save(model_state, model_filepath)
    if isbest:
        logger.info("Saving best checkpoint copy")
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth'))
        shutil.copyfile(model_filepath, os.path.join(checkpoint, 'model_best.pth'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Load checkpoint file into model (and optimizer if provided).

    The key remapping logic below is kept for compatibility with older
    checkpoint formats used during project development.
    """
    if not os.path.exists(checkpoint):
        raise IOError("File doesn't exist {}".format(checkpoint))

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')

    state_dict = {}
    for key in checkpoint['state_dict'].keys():
        if 'layers.0.' in key:
            state_dict[key.split('0.')[0].split('module.')[1] + key.split('0.')[1]] = checkpoint['state_dict'][key]
        elif 'layers.1.' in key:
            state_dict[key.replace('1', '8').split('module.')[1]] = checkpoint['state_dict'][key]
        elif 'module.' in key:
            state_dict[key.split('module.')[1]] = checkpoint['state_dict'][key]
        else:
            state_dict[key] = checkpoint['state_dict'][key]
    model.load_state_dict(state_dict)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

