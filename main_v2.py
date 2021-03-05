import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import os
import cv2
import timm


from utils import get_config, get_data, get_device, get_transform, compute_class_freqs, train, test, get_current_dir

cfg = get_config()


class DatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, df, transform):

        self.transform = transform
        self.df = df
        self.labels = self.df[cfg["TARGET_COLS"]].values
        self.file_names = df[cfg["IMAGE_COL"]].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.file_names[idx]
        image_path = f'{cfg["DATASET_DIR"]}/train/{image_name}.jpg'
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image_t = self.transform(image)
        label = self.labels[idx]
        data = {
            "image": image_t,
            "targets": torch.tensor(self.labels[idx]).float()
        }

        return data


class TLModel(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super(TLModel).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        for param in self.model.parameters():
            param.requires_grad = False
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, cfg["TARGET_SIZE"])

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x


class ModelCheckpoint:
    def __init__(self, model, filename=None, filepath=None):
        self.min_loss = None
        self.model = model

        if filepath is None:
            filepath = os.path.join(get_current_dir(), "Model Saved")

        if filename is None:
            filename = "best_model.pt"

        self.filepath = os.path.join(filepath, filename)

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print(f"\tSaving a better model here: {self.filepath}", end='\n')
            torch.save(self.model, self.filepath)
            self.min_loss = loss


def main():
    device = get_device()

    data = get_data()
    train_data = data.sample(frac=cfg["TRAIN_VALIDATION_FRAC"])
    validation_data = data.drop(train_data.index)

    transform = get_transform(image_size=cfg["IMAGE_SIZE"])

    train_dataset = DatasetTransformer(train_data, transform=transform)
    validation_dataset = DatasetTransformer(
        validation_data, transform=transform)

    train_dataset_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True)
    validation_dataset_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True)

    model = TLModel(cfg["MODEL_NAME"], pretrained=True)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=20, gamma=0.5)
    loss = nn.BCELoss()

    model_checkpoint = ModelCheckpoint(model)

    freq_pos, freq_neg = compute_class_freqs(train_data.iloc[:, 1:12])

    print(f"Training on {len(train_dataset)} images")

    for t in range(cfg["EPOCHS"]):
        print(f'Epoch: {t}')
        scheduler.step()

        train_loss, train_auc = train(model=model, data_loader=train_dataset_loader, loss_function=loss,
                                      optimizer=optimizer, weight_pos=freq_pos, weight_neg=freq_neg, device=device)

        print(
            f'\tTraining step: Loss: {train_loss}, AUC: {train_auc}', end='\n')

        val_loss, val_auc = test(model=model, data_loader=validation_dataset_loader,
                                 loss_function=loss, weight_pos=freq_pos, weight_neg=freq_neg, device=device)
        print(f'\tValidation step: Loss: {val_loss}, AUC: {val_auc}', end='\n')

        torch.save(model.state_dict(), f'checkpoint_epoch_{t}.pth')
        print('\tModel saved')

        model_checkpoint.update(loss=val_loss)


if __name__ == "__main":
    main()
