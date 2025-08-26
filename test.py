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
from monai.inferers import sliding_window_inference

def main():
    ##### Load hyperparameters from YAML file
    with open('configs.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        
    ##### dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _,val_ds, _,_ = data_loaders(config, device=device)

    ##### define the model
    model = MAFTCNet(
        img_size=(config["input_size"], config["input_size"], config["input_size"]),
        in_channels=config["input_channels"],
        out_channels=config["num_classes"],
        feature_size=config["feature_size"],
        use_checkpoint=config["use_checkpoint"],
    ).to(device)

    root_dir = config["saved_model_dir"]
    
    slice_map = {
        "img0029.nii.gz": 170,
        "img0030.nii.gz": 230,
        "img0031.nii.gz": 70,
        "img0032.nii.gz": 204,
        "img0033.nii.gz": 204,
        "img0034.nii.gz": 180,
    }     
    case_num = 2
    print("Stat loading the model!")
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_model.pth")))
    print("Model loaded! ")
    model.eval()
    with torch.no_grad():
        img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        val_inputs = torch.unsqueeze(img, 1).cuda()
        val_labels = torch.unsqueeze(label, 1).cuda()
        val_outputs = sliding_window_inference(val_inputs, (config["input_size"], config["input_size"], config["input_size"]), 1, model, overlap=0.8)

        # --- Plot three subplots ---
        plt.figure("check", (18, 6))

        plt.subplot(1, 3, 1)
        plt.title("image")
        print("img_name-------------", img_name)
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")

        plt.subplot(1, 3, 2)
        plt.title("label")
        plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])

        plt.subplot(1, 3, 3)
        plt.title("output")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]])

        # --- Save figure instead of showing ---
        save_path = os.path.join(config["save_dir"], f"{img_name}_check.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

        print(f"Saved visualization to: {save_path}")


if __name__ == "__main__":
    main()
