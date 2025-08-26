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
import argparse

def save_checkpoint(path, model, global_step, dice_val_best, global_step_best):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "global_step": global_step,
            "dice_val_best": dice_val_best,
            "global_step_best": global_step_best,
        },
        path,
    )
    print(f"[ckpt] Saved checkpoint -> {path}")

def load_checkpoint(path, model, device):
    print(f"[ckpt] Loading checkpoint from: {path}")
    ckpt = torch.load(path, map_location=device)

    # Case 1: checkpoint is a dict with 'model_state_dict'
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        global_step = ckpt.get("global_step", 0)
        dice_val_best = ckpt.get("dice_val_best", 0.0)
        global_step_best = ckpt.get("global_step_best", 0)
    else:
        # Case 2: checkpoint is just the state_dict
        model.load_state_dict(ckpt, strict=True)
        global_step, dice_val_best, global_step_best = 0, 0.0, 0

    print(
        f"[ckpt] Model loaded. step={global_step}, "
        f"dice_val_best={dice_val_best:.5f}, best_step={global_step_best}"
    )
    return global_step, dice_val_best, global_step_best

def main():
    # -----------------------
    # CLI
    # -----------------------
    parser = argparse.ArgumentParser(description="Train MAFTCNet")
    parser.add_argument(
        "resume",
        nargs="?",
        default=None,
        help="Optional path to a checkpoint .pth to resume from",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (overrides configs.yaml if set)",
    )
    args = parser.parse_args()

    # -----------------------
    # Load config
    # -----------------------
    with open("configs.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader, _, train_loader, _ = data_loaders(config, device=device)

    # -----------------------
    # Model
    # -----------------------
    model = MAFTCNet(
        img_size=(config["input_size"], config["input_size"], config["input_size"]),
        in_channels=config["input_channels"],
        out_channels=config["num_classes"],
        feature_size=config["feature_size"],
        use_checkpoint=config["use_checkpoint"],
    ).to(device)

    # -----------------------
    # Checkpoint setup
    # -----------------------
    ckpt_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else config.get("saved_model_dir", "./checkpoints")
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt_path = os.path.join(ckpt_dir, "last.pth")

    # -----------------------
    # Initialize / Resume
    # -----------------------
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0

    if args.resume is not None and os.path.isfile(args.resume):
        global_step, dice_val_best, global_step_best = load_checkpoint(
            args.resume, model, device
        )
    else:
        print("[ckpt] No resume path provided; training from scratch.")

    # -----------------------
    # Train loop
    # (Note: if your utils.train already saves 'best' checkpoints,
    #  this script will coexist with that. Here we save 'last' at the end.)
    # -----------------------
    while global_step < config["max_iterations"]:
        global_step, dice_val_best, global_step_best = train(
            model,
            global_step,
            train_loader,
            val_loader,
            config,
            dice_val_best,
            global_step_best,
        )

    # Save final/last checkpoint (model weights + counters)
    save_checkpoint(last_ckpt_path, model, global_step, dice_val_best, global_step_best)

    # Also save a step-tagged checkpoint for reproducibility
    tagged_path = os.path.join(ckpt_dir, f"model_step_{global_step}.pth")
    save_checkpoint(tagged_path, model, global_step, dice_val_best, global_step_best)

if __name__ == "__main__":
    main()
