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


def get_config():
    cfg = {
        "BATCH_SIZE": 32,
        "EPOCHS": 10,
        "IMAGE_SIZE": 528,
        "TARGET_SIZE": 11,
        "TARGET_COLS": ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                        'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
                        'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                        'Swan Ganz Catheter Present'],
        "IMAGE_COL": "StudyInstanceUID",
        "DATASET_DIR": "../input/ranzcr-clip-catheter-line-classification/",
        "MODEL_NAME": "efficientnet_b6",
        "LR": 0.01,
        "WEIGHT_DECAY": 0.0,
        "TRAIN_VALIDATION_FRAC": 0.8

    }
    return cfg


def get_data(mode="test", percentage=0.3):
    data = pd.read_csv(
        '/kaggle/input/ranzcr-clip-catheter-line-classification/train.csv')
    if mode != "training":
        data = data.sample(frac=percentage)
    else:
        percentage = 1
    print(f"{mode} mode used, percentage of the data used: {percentage*100}%")
    return data.sample(frac=percentage)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_transform(image_size, augmented=False):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
    ])


def compute_class_freqs(labels):

    labels = np.array(labels)

    N = labels.shape[0]

    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies

    return positive_frequencies, negative_frequencies


def weighted_loss(pos_weights, neg_weights, y_pred, y_true, epsilon=1e-7):

    loss = 0.0
    for i in range(len(pos_weights)):
        loss_pos = -1 * \
            torch.mean(pos_weights[i] * y_true[:, i] *
                       torch.log(y_pred[:, i] + epsilon))
        loss_neg = -1 * \
            torch.mean(neg_weights[i] * (1-y_true[:, i])
                       * torch.log((1-y_pred[:, i]) + epsilon))
        loss += loss_pos + loss_neg

    return loss


def mean_roc_auc(targets, ouputs, target_size=11):
    roc_auc = []
    for k in range(target_size):
        try:
            roc_auc.append(metrics.roc_auc_score(targets[:, k], ouputs[:, k]))
        except Exception as e:
            roc_auc.append(0.5)
            pass
    return np.mean(roc_auc)


def train(model, data_loader, loss_function, optimizer, weight_pos, weight_neg, device):
    """
        It trains the model for one epoch
    :param model: model we need to train
    :param data_loader: Data loader (iterable)
    :param loss_function: Loss Function
    :param optimizer: Optimizer (e.g Adam)
    :param device: "cpu" or "cuda"
    :return: None
    """
    total_number = 0.0
    total_loss = 0.0
    mean_auc = []
    model.train()
    for _, data in enumerate(data_loader):
        image_input = data["image"].to(device)
        targets = data["targets"].to(device)
        outputs = model(image_input)

        loss = loss_function(outputs, targets)

        total_number += image_input.shape[0]
        total_loss += image_input.shape[0] * loss.item()
        mean_auc.append(mean_roc_auc(
            targets.detach().cpu(), outputs.detach().cpu()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / total_number, np.average(mean_auc)


def test(model, data_loader, loss_function, weight_pos, weight_neg, device):
    with torch.no_grad():
        model.eval()
        total_number = 0
        total_loss = 0.0
        mean_auc = []
        for _, data in enumerate(data_loader):
            image_input = data["image"].to(device)
            targets = data["targets"].to(device)
            outputs = model(image_input)
            total_number += image_input.shape[0]

            total_loss += image_input.shape[0] * \
                loss_function(outputs, targets).item()
            mean_auc.append(mean_roc_auc(targets.cpu(), outputs.cpu()))
    return total_loss / total_number, np.average(mean_auc)


def get_current_dir():
    return os.path.dirname(os.path.realpath(__file__))


if __name__ == "__main__":
    pass
