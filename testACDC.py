import torch
import os
import matplotlib.pyplot as plt
import yaml
import argparse
from monai.inferers import sliding_window_inference
from model.MAFTCNet import MAFTCNet
from loader.loaderACDC import data_loaders

def main():
    # -------------------
    # CLI arguments
    # -------------------
    parser = argparse.ArgumentParser(description="Test MAFTCNet")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the trained model checkpoint (.pth)"
    )
    args = parser.parse_args()

    # -------------------
    # Load config
    # -------------------
    with open("configsSynapse.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_ds, _, _ = data_loaders(config, device=device)

    # -------------------
    # Define model
    # -------------------
    model = MAFTCNet(
        img_size=(config["input_size"], config["input_size"], config["input_size"]),
        in_channels=config["input_channels"],
        out_channels=config["num_classes"],
        feature_size=config["feature_size"],
        use_checkpoint=config["use_checkpoint"],
    ).to(device)

    # -------------------
    # Load checkpoint
    # -------------------
    print(f"[ckpt] Loading model from: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print("[ckpt] Model loaded successfully.")

    model.eval()

    # -------------------
    # Example visualization
    # -------------------
    slice_map = {
        "patient132_frame10.nii": 180,
        "patient147_frame09.nii": 180,
        "patient167_frame08.nii": 180,
    }
    case_num = 2

    with torch.no_grad():
        img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]

        val_inputs = torch.unsqueeze(img, 1).to(device)
        val_labels = torch.unsqueeze(label, 1).to(device)

        val_outputs = sliding_window_inference(
            val_inputs,
            (config["input_size"], config["input_size"], config["input_size"]),
            1,
            model,
            overlap=0.8,
        )

        # --- Plot three subplots ---
        plt.figure("check", (18, 6))

        plt.subplot(1, 3, 1)
        plt.title("image")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")

        plt.subplot(1, 3, 2)
        plt.title("label")
        plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])

        plt.subplot(1, 3, 3)
        plt.title("output")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]])

        # --- Save figure instead of showing ---
        os.makedirs(config["save_dir"], exist_ok=True)
        save_path = os.path.join(config["save_dir"], f"{img_name}_check.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

        print(f"[viz] Saved visualization to: {save_path}")


if __name__ == "__main__":
    main()
