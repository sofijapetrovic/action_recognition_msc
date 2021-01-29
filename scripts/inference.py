from dataset import PoseDataset
from model import PoseNet
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from os import listdir
from eval import eval

if __name__ == '__main__':
    checkpoint_folder = 'checkpoints/'
    cps = sorted([c for c in listdir(checkpoint_folder)], key=lambda name: int(name.split("_")[1][:-4]), reverse=True)
    print(cps)
    for cp in cps:
        print(cp)
        eval(os.path.join(checkpoint_folder,cp), None)
