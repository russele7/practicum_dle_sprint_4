
import re
from random import shuffle

from PIL import Image
import albumentations as A

from torch.utils.data import Dataset
import timm
from transformers import AutoTokenizer
from src.constants import DATA_PATH
import torch


def text_interfusion(text):
    text_list = text.split(", ")
    shuffle(text_list)
    return ", ".join(text_list) 


def texts_interfusion(texts):
    texts_lists = [t.split(", ") for t in texts]
    for tl in texts_lists:
        shuffle(tl)
    return [", ".join(t) for t in texts_lists]


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.RandomCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Affine(scale=(0.8, 1.2),
                        rotate=(-15, 15),
                        translate_percent=(-0.1, 0.1),
                        shear=(-10, 10),
                        fill=0,
                        p=0.8),
                A.CoarseDropout(num_holes_range=(2, 8),
                                hole_height_range=(int(0.07 * cfg.input_size[1]),
                                                int(0.15 * cfg.input_size[1])),
                                hole_width_range=(int(0.1 * cfg.input_size[2]),
                                                int(0.15 * cfg.input_size[2])),
                                fill=0,
                                p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.CenterCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ]
        )

    return transforms


def set_requires_grad(module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for param, _ in module.named_parameters():
            param.requires_grad = False
        return

    pattern = re.compile(unfreeze_pattern)

    for name, param in module.named_parameters():
        if pattern.search(name):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False