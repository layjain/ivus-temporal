import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import utils
from models.registration_model import RegistrationModel
from data.ants_data import ANTsDataset
import registration

import time

DATAFRAME = []

def get_dataloader(args):
    

def main(args):
    model = RegistrationModel(args)
    print(f"Loaded model with {utils.count_parameters(model) / 10**6}M parameters")
    model = model.to(args.device)

    vis = utils.visualize.Visualize(args) if args.visualize else None
    mnist_dataloader = get_dataloader(args)