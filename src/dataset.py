
import re
import random

import pandas as pd
from PIL import Image
import albumentations as A
import numpy as np
from torch.utils.data import Dataset
import timm
from transformers import AutoTokenizer
from src.constants import DATA_PATH
import torch


class MultimodalDataset(Dataset):
    def __init__(self,
                 config,
                 transforms,
                 mode,):
        if mode not in ['train', 'test']:
            raise ValueError(f'mode = {mode}')
        df = pd.read_csv(f'{DATA_PATH}dish.csv')
        self.df = df[df['split'] == mode].reset_index(drop=True)

        self.df = df
        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.transform = transforms

        df_ing = pd.read_csv(f'{DATA_PATH}ingredients.csv')
        self.dict_ingredients = {k: v for k, v in
                                 zip(df_ing['id'], df_ing['ingr'])}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # ingredients_idx = 
        ingredients_idx = get_ingredients_idx(self.df.loc[idx, "ingredients"])

        text = ', '.join([self.dict_ingredients[idx]
                          for idx in ingredients_idx])

        dish_id = self.df.loc[idx, "dish_id"]
        img_path = f'{DATA_PATH}images/{dish_id}/rgb.png'
        image = Image.open(img_path).convert('RGB')
        # print(f'type(image) = {type(image)}')
        # print(f'image.shape = {image.shape}')
        image = self.transform(image=np.array(image))["image"]

        total_mass = self.df.loc[idx, "total_mass"]

        total_calories = self.df.loc[idx, "total_calories"]

        return {
            "target": total_calories,
            "image": image,
            # "input_ids": tokenized_input["input_ids"],
            # "attention_mask": tokenized_input["attention_mask"],
            "text": text,
            "total_mass": total_mass,
        }
    
def get_ingredients_idx(raw_text):
    return [int(t[12:]) for t in raw_text.split(';')]

    
def text_interfusion(text):
    text_list = text.split(", ")
    random.shuffle(text_list)
    return ", ".join(text_list) 


def texts_interfusion(texts):
    texts_lists = [t.split(", ") for t in texts]
    for tl in texts_lists:
        random.shuffle(tl)
    return [", ".join(t) for t in texts_lists]


def collate_fn(batch, tokenizer, mode):
    texts = [item["text"] for item in batch]

    if mode == 'train':
        texts = [text_interfusion(text) for text in texts]

    total_masses = torch.LongTensor([item["total_mass"] for item in batch])

    images = torch.stack([item["image"] for item in batch])

    targets = torch.LongTensor([item["target"] for item in batch])    

    tokenized_input = tokenizer(texts,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True)
    return {
        "target": targets,
        "image": images,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"],
        "total_mass": total_masses,
    }



def get_transforms(config, mode="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if mode == "train":
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
