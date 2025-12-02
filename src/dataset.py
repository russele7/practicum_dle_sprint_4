
import re

from PIL import Image
import albumentations as A

from torch.utils.data import Dataset
import timm
from transformers import AutoTokenizer
from src.constants import DATA_PATH
import torch


class MultimodalDataset(Dataset):
    def __init__(self,
                 df,
                 text_model,
                 image_model,
                 transforms,
                 dict_ingredients):
        self.df = df
        self.image_cfg = timm.get_pretrained_cfg(image_model)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.transform = transforms
        self.dict_ingredients = dict_ingredients

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):        

        ingredients_idx = self.df.loc[idx, "ingredients_idx"]
        text = ', '.join([self.dict_ingredients[idx]
                          for idx in ingredients_idx])

        dish_id = self.df.loc[idx, "dish_id"]
        img_path = f'{DATA_PATH}images/{dish_id}/rgb.png'
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

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
    
def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]

    # if text_augmentation:
    #texts = [text_interfusion(text) for text in texts]

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
