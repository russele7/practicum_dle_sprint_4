
import os
import re
import random
from random import shuffle, sample
from functools import partial

from PIL import Image
from matplotlib import pyplot as plt
import albumentations as A
import numpy as np
import timm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
# from src.constants import DATA_PATH
from tqdm import tqdm
import torchmetrics
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer

from .dataset import MultimodalDataset, get_transforms, collate_fn
from .constants import DATA_PATH


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


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


class MultimodalModel(nn.Module):
    def __init__(self, config,):
        super().__init__()
        # self.hidden_state = emb_dim
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.regressor = nn.Sequential(
            nn.Sequential(
                nn.Linear(
                    in_features=(
                        self.text_model.config.hidden_size
                        + self.image_model.num_features
                        + 1
                    ),
                    out_features=config.HIDDEN_DIM,
                    bias=True
                ),
                nn.LayerNorm(config.HIDDEN_DIM),
                nn.Hardswish(),
                nn.Dropout(p=config.DROPOUT),
                nn.Linear(in_features=config.HIDDEN_DIM,
                          out_features=1, bias=True),
            )
        )

    def forward(self, text_input, image_input, mass_input):
        text_features = self.text_model(
            **text_input).last_hidden_state[:,  0, :]
        image_features = self.image_model(image_input)
        mass_features = torch.tensor(mass_input).unsqueeze(1)

        # print(text_features.shape, image_features.shape, mass_features.shape)
        multi_output = torch.cat(
            [text_features, image_features, mass_features], dim=1)
        # print(multi_output.shape)
        output = self.regressor(multi_output)
        # print(output.shape)
        return output


def train(config, device):
    seed_everything(config.SEED)

    # Инициализация модели
    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    set_requires_grad(model.text_model,
                      unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model,
                      unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    # Оптимизатор с разными LR
    optimizer = AdamW([{
        'params': model.text_model.parameters(),
        'lr': config.TEXT_LR
    }, {
        'params': model.image_model.parameters(),
        'lr': config.IMAGE_LR
    }, {
        'params': model.regressor.parameters(),
        'lr': config.REGRESSOR_LR
    }])

    criterion = nn.L1Loss()

    # Загрузка данных
    transforms = get_transforms(config, mode="train")
    val_transforms = get_transforms(config, mode="test")
    train_dataset = MultimodalDataset(config, transforms, mode="train")
    val_dataset = MultimodalDataset(config, val_transforms, mode="test")
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 tokenizer=tokenizer,
                                                 mode="train"))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn,
                                               tokenizer=tokenizer,
                                               mode="test"))

    # инициализируем метрику
    torchmetrics.MeanAbsoluteError()

    mae_metric_train = torchmetrics.MeanAbsoluteError().to(device)
    mae_metric_val = torchmetrics.MeanAbsoluteError().to(device)
    # best_mae_train = 0.0
    best_mae_val = 0.0

    print("training started")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        print(f'epoch {epoch}')
        for batch in tqdm(train_loader):
            # Подготовка данных
            # inputs = {
            #     'input_ids': batch['input_ids'].to(device),
            #     'attention_mask': batch['attention_mask'].to(device),
            #     'image': batch['image'].to(device),
            #     'total_mass': batch['total_mass'].to(device)
            # }
            inputs = {
                'text_input': {'input_ids': batch['input_ids'].to(device),
                               'attention_mask': batch['attention_mask'].to(device)},
                'image_input': batch['image'].to(device),
                'mass_input': batch['total_mass'].to(device)
            }
            # labels = batch['label'].to(device)
            targets = batch['target'].to(device)

            # forward(self, text_input, image_input, mass_input)


            # Forward
            optimizer.zero_grad()
            preds = model(**inputs).squeeze()
            loss = criterion(preds, targets)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _ = mae_metric_train(preds=preds, target=targets)

        # Валидация
        train_mae = mae_metric_train.compute().cpu().numpy()
        val_mae = validate(model, val_loader, device, mae_metric_val)
        mae_metric_val.reset()
        mae_metric_train.reset()

        print(
            f"Epoch {epoch}/{config.EPOCHS-1} | "
            f"avg_Loss: {total_loss/len(train_loader):.4f} | "
            f"Train MAE: {train_mae :.4f}| "
            f"Val MAE: {val_mae :.4f}"
        )

        if val_mae > best_mae_val:
            print(f"New best model, epoch: {epoch}")
            best_mae_val = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)


def validate(model, val_loader, device, metric):
    model.eval()

    with torch.no_grad():
        for batch in tqdm(val_loader):

            # inputs = {
            #     'input_ids': batch['input_ids'].to(device),
            #     'attention_mask': batch['attention_mask'].to(device),
            #     'image': batch['image'].to(device),
            #     'total_mass': batch['total_mass'].to(device)
            # }
            inputs = {
                'text_input': {'input_ids': batch['input_ids'].to(device),
                               'attention_mask': batch['attention_mask'].to(device)},
                'image_input': batch['image'].to(device),
                'mass_input': batch['total_mass'].to(device)
            }

            targets = batch['target'].to(device)

            preds = model(**inputs).squeeze()
            _ = metric(preds=preds, target=targets)

    return metric.compute().cpu().numpy()


def plot_sample_images(dish_ids):

    image_paths = [f'{DATA_PATH}images/{d}/rgb.png' for d in dish_ids]

    _, axs = plt.subplots(3, 3, figsize=(5, 5))
    axs = axs.flatten()
    image_paths_sample = sample(image_paths, 9)
    imgs = [Image.open(img_path).convert('RGB') for img_path in image_paths_sample]
    for img, ax in zip(imgs, axs):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img)


def plot_images_flatten(dish_ids, sample_flg=True, sample_count=5, figsize=(8, 8),):

    image_paths = [f'{DATA_PATH}images/{d}/rgb.png' for d in dish_ids]
    if sample_flg:
        _, axs = plt.subplots(1, sample_count, figsize=figsize)
        image_paths_sample = sample(image_paths, sample_count)
    else:
        _, axs = plt.subplots(1, len(image_paths), figsize=figsize)
        image_paths_sample = image_paths
    axs = axs.flatten()    
    imgs = [Image.open(img_path).convert('RGB') for img_path in image_paths_sample]
    for img, ax in zip(imgs, axs):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img)