import torch
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.losses import DiceCELoss
from utils import train
from model.MAFTCNet import MAFTCNet 
from loader.loader import data_loaders
from thop import profile
import time 
import yaml

def main():
    ##### Load hyperparameters from YAML file
    with open('configs.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        
    ##### dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader,_, train_loader,_ = data_loaders(config, device=device)

    ##### define the model
    model = MAFTCNet(
        img_size=(config["input_size"], config["input_size"], config["input_size"]),
        in_channels=config["input_channels"],
        out_channels=config["num_classes"],
        feature_size=config["feature_size"],
        use_checkpoint=config["use_checkpoint"],
    ).to(device)

    ##### train loop 
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    while global_step < config["max_iterations"]:
        global_step, dice_val_best, global_step_best = train(model, global_step, train_loader, val_loader,config, dice_val_best, global_step_best)

if __name__ == "__main__":
    main()
