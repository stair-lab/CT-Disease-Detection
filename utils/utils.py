import random
import torch
import os
import numpy as np
from argparse import ArgumentParser
import json


def init(args):

    seed = args.seed

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'

    return


def arg_parse():

    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='data', help='Directory with data')
    parser.add_argument('--checkpoint_dir', default='checkpoints', help='Checkpoint output directory')
    parser.add_argument('--pretrain', default=None, help='Path to pretrained weights')
    parser.add_argument('--size', default=256, type=int, help="Size of CT to generate")
    parser.add_argument('--age_norm', default=100.0, type=float, help="Normalization of age")
    parser.add_argument('--raf_norm', default=50.0, type=float, help="Normalization of RAF")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('--epochs', default=1000, type=int, help="Number of epochs")
    parser.add_argument('--train_batch_size', default=64, type=int, help="Training batch size")
    parser.add_argument('--val_batch_size', default=64, type=int, help="Validation batch size")
    parser.add_argument('--test_batch_size', default=64, type=int, help="Testing batch size")
    parser.add_argument('--num_workers', default=32, type=int, help="Number of workers used")
    parser.add_argument('--decay_start_epoch', default=15, type=int, help="Epoch to start learning rate decay")
    parser.add_argument('--seed', default=42, type=int, help="Seed")
    parser.add_argument('--two_view', default=False, type=bool, help="One-view CT or Two-view CT")
    parser.add_argument('--early_stop_criteria', default=20, type=int, help="Early stopping criteria (# of epochs)")

    args = parser.parse_args()
    print('Command line arguments:')
    print(json.dumps(vars(args), indent=4))

    return args
