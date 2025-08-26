from monai.transforms import MapTransform
import os
import shutil
import tempfile
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)

class ConvertLabelsToZero(MapTransform):
    """
    This custom transformation will convert specified labels to zero.
    """
    def __init__(self, keys, labels_to_zero):
        super().__init__(keys)
        self.labels_to_zero = labels_to_zero

    def __call__(self, data):
        for key in self.keys:
            for label in self.labels_to_zero:
                data[key][data[key] == label] = 0
        return data

class NormalizeArray(MapTransform):
    """
    This custom transformation will normalize the array data to the [0, 1] range.
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            arr = data[key]
            # Calculate the min and max, considering the array shape [B, C, D, H, W]
            min_val = arr.min()
            max_val = arr.max()
            # Perform min-max normalization
            data[key] = (arr - min_val) / (max_val - min_val)
        return data


def data_transformers(config, device):
  train_transforms = Compose(
      [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        NormalizeArray(keys=["image"]),

        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim= config["pixdim"],
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        ConvertLabelsToZero(keys=["label"], labels_to_zero=[5,9,10,12,13]),  # remove small organs
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(config["input_size"], config["input_size"], config["input_size"]),
            pos=1,
            neg=1,
            num_samples=config["num_samples"],
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob= config["prob_RandFlipd"],
            ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob= config["prob_RandFlipd"],
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob= config["prob_RandFlipd"],
            ),
        RandRotate90d(
            keys=["image", "label"],
            prob=config["prob_RandRote"],
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=config["offset_ShiftInten"],
            prob=config["prob_ShiftInten"], 
        ),
      ]
  )
  val_transforms = Compose(
      [
          LoadImaged(keys=["image", "label"], ensure_channel_first=True),
          NormalizeArray(keys=["image"]),
          CropForegroundd(keys=["image", "label"], source_key="image"),
          Orientationd(keys=["image", "label"], axcodes="RAS"),
          Spacingd(
              keys= ["image", "label"],
              pixdim= config["pixdim"],
              mode=("bilinear", "nearest"),
          ),
          EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
      ]
  )
  return train_transforms, val_transforms


def data_loaders(config, device):
  split_json = "dataset.json"
  datasets = config["data_dir"] + split_json
  train_transforms, val_transforms = data_transformers(config, device)
  datalist = load_decathlon_datalist(datasets, True, "training")
  val_files = load_decathlon_datalist(datasets, True, "validation")
  train_ds = CacheDataset(
      data=datalist,
      transform=train_transforms,
      cache_num=config["train_cache_num"],
      cache_rate=config["cache_rate"],
      num_workers=config["train_num_workers"],
  )
  train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=config["batch_size"], shuffle=True)
  val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=config["val_cache_num"], cache_rate=1.0, num_workers=config["val_num_workers"])
  val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=config["batch_size"])
  set_track_meta(False)
  return val_loader,val_ds, train_loader, train_ds



