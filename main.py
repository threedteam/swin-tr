import sys
import os
import os
import sys
import time
import random
import string
import argparse
import re
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
import copy
from utils.utils import Averager, TokenLabelConverter, get_args, draw_one_loss, draw_one_acc, draw_mul_loss, draw_mul_acc
from data.dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from modules.model import Model
from test import validation
import utils.utils_dist as utils

# loss calculator for LISTER NeighbourDecoder Module
lister_loss = utils.ListerLoss()

import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

import torch
import PIL

# evaluation statistics utilities
from utils import results_statistics

import matplotlib.pyplot as plt


def main():
    from utils.utils import get_args

    if len(sys.argv) < 2:
        print("Usage:")
        print("  For training: python main.py --train_data [path] --valid_data [path] [other options]")
        print("  For testing:  python main.py --eval_data [path] --saved_model [path] [other options]")
        return

    # Check if training or testing mode
    is_train = '--eval_data' not in sys.argv

    if is_train:
        print("Starting training...")
        from train import train
        opt = get_args()
        if not opt.exp_name:
            opt.exp_name = f'{opt.TransformerModel}' if opt.Transformer else f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        os.makedirs(f'{opt.saved_path}/{opt.exp_name}', exist_ok=True)

        """ vocab / character number configuration """
        if opt.sensitive:
            # Explicitly exclude whitespace characters (spaces, newlines, etc.)
            opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
        utils.init_distributed_mode(opt)
        seed = opt.manualSeed + utils.get_rank()  
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # Common PyTorch performance flags
        cudnn.benchmark = True
        cudnn.deterministic = True
        opt.num_gpu = torch.cuda.device_count()
        train(opt)
    else:
        print("Starting testing...")
        from test import test
        opt = get_args(is_train=False)

        """ vocab / character number configuration """
        if opt.sensitive:
            opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        cudnn.benchmark = True
        cudnn.deterministic = True
        opt.num_gpu = torch.cuda.device_count()

        # Skip tabulate-based formatting if tabulate is not used
        if opt.range is not None:
            start_range, end_range = sorted([int(e) for e in opt.range.split('-')])
            print("eval range: ",start_range,end_range)
        opt.saved_model = opt.model_dir
        test(opt)


if __name__ == '__main__':
    main()