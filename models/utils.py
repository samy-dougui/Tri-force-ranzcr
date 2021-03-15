import numpy as np
import pandas as pd
import torchvision.datasets as datasets
import torch
import torch.nn as nn
from sklearn import metrics
import os
import cv2
import timm
import matplotlib.pyplot as plt
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchsummary import summary

from models import EfficientNet, Resnet, Ensemble


def get_config():
    return {
        "BATCH_SIZE": 32,
        "EPOCHS": 10,
        "IMAGE_SIZE": {
            "efficientnet": 260,
            "resnet": 256
        },
        "TARGET_SIZE": 11,
        "models": {
            "resnet": "./models/resnet200d.pth",
            "efficientnet": "./models/efficientnetb2.pth"
        },
        "TARGET_COLS": ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                        'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
                        'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                        'Swan Ganz Catheter Present'],
        "IMAGE_COL": "StudyInstanceUID",
        "DATASET_DIR": "./input/ranzcr-clip-catheter-line-classification/",
        "MODEL_NAME": "efficientnet_b7",
        "LR": 0.01,
        "WEIGHT_DECAY": 0.0,
        "TRAIN_VALIDATION_FRAC": 0.9

    }


class DatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, df, transform, cfg):

        self.transform = transform
        self.df = df
        self.cfg = cfg
        self.labels = self.df[self.cfg["TARGET_COLS"]].values
        self.file_names = df[self.cfg["IMAGE_COL"]].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.file_names[idx]
        image_path = f'{self.cfg["DATASET_DIR"]}/train/{image_name}.jpg'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_t = self.transform(image=image)["image"]
        data = {
            "image": image_t.float(),
            "targets": torch.tensor(self.labels[idx]).float()
        }

        return data


class DatasetTransformer_prediction(torch.utils.data.Dataset):
    def __init__(self, df, cfg):

        self.df = df
        self.cfg = cfg
        self.transform1 = A.Compose(
            [
                A.Resize(self.cfg["IMAGE_SIZE"]["efficientnet"],
                         self.cfg["IMAGE_SIZE"]["efficientnet"]),
                ToTensorV2()
            ])
        self.transform2 = A.Compose(
            [
                A.Resize(self.cfg["IMAGE_SIZE"]["resnet"],
                         self.cfg["IMAGE_SIZE"]["resnet"]),
                ToTensorV2()
            ])
        self.file_names = self.df[self.cfg["IMAGE_COL"]].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.file_names[idx]
        image_path = f'{self.cfg["DATASET_DIR"]}/test/{image_name}.jpg'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_t_1 = self.transform1(image=image)['image']
        image_t_2 = self.transform2(image=image)['image']
        data = {
            "image_efficientnet": image_t_1.float(),
            "image_resnet": image_t_2.float(),
        }

        return data


class ModelCheckpoint:
    def __init__(self, model, filepath=None, filename=None):
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
            torch.save(self.model.state_dict(), self.filepath)
            self.min_loss = loss


def get_data(mode="test", percentage=0.3):
    try:
        data = pd.read_csv(
            './input/ranzcr-clip-catheter-line-classification/train.csv')
    except Exception:
        print("Please put the training dataset here: ./input/ranzcr-clip-catheter-line-classification/train.csv")
        return

    if mode != "training":
        print(f"[{mode}] mode: Using only {percentage}% of the data")
        return data.sample(frac=percentage)

    print("[Training] mode, using all the data")
    return data


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # if torch.cuda.is_available():
    #     return torch.device("cuda")
    # return torch.device("cpu")


def get_transform(image_size, model_name, cfg, augmented=False):
    if augmented:
        if model_name == "efficientnet":
            return A.Compose([
                A.Resize(cfg["IMAGE_SIZE"][model_name],
                         cfg["IMAGE_SIZE"][model_name]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2()
            ])
        elif model_name == "resnet":
            return A.Compose([
                A.RandomResizedCrop(
                    cfg["IMAGE_SIZE"][model_name], cfg["IMAGE_SIZE"][model_name], scale=(0.9, 1), p=1),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            print("Error returning the transformation, returning the nan-augmented one")

    return A.Compose([
        A.Resize(image_size, image_size),
        ToTensorV2()
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
        except Exception:
            roc_auc.append(0.5)
    return np.mean(roc_auc)


def train(model, data_loader, loss_function, optimizer, device):
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


def test(model, data_loader, loss_function, device):
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


def main_train(cfg, model_name, verbose=False):
    device = get_device()

    data = get_data()
    if data:
        train_data = data.sample(frac=cfg["TRAIN_VALIDATION_FRAC"])
        validation_data = data.drop(train_data.index)
    else:
        return

    train_transform = get_transform(
        image_size=cfg["IMAGE_SIZE"][model_name], model_name=model_name, cfg=cfg, augmented=True)
    validation_transform = get_transform(
        image_size=cfg["IMAGE_SIZE"][model_name], model_name=model_name, cfg=cfg,)

    train_dataset = DatasetTransformer(
        train_data, transform=train_transform, cfg=cfg)
    validation_dataset = DatasetTransformer(
        validation_data, transform=validation_transform, cfg=cfg)

    train_dataset_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True)
    validation_dataset_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True)

    if model_name == "resnet":
        model = Resnet(cfg=cfg, pretrained=True)
        pass
    elif model_name == "efficientnet":
        model = EfficientNet(cfg=cfg, pretrained=True)
    else:
        print("Wrong model name")
        return

    model.to(device)

    if verbose:
        summary(model, (3, cfg["IMAGE_SIZE"], cfg["IMAGE_SIZE"]))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, patience=3, factor=0.5, verbose=verbose, mode='min', threshold=2*1e-2)

    loss = nn.BCELoss()

    train_updater = GraphUpdater(type="Train")
    validation_updater = GraphUpdater(type="Validation")
    model_checkpoint = ModelCheckpoint(model)

    print(f"Training on {len(train_dataset)} images")

    for t in range(cfg["EPOCHS"]):
        print(f'Epoch: {t}')

        train_loss, train_auc = train(model=model, data_loader=train_dataset_loader, loss_function=loss,
                                      optimizer=optimizer, device=device)

        train_updater.update(**{"loss": train_loss, "accuracy": train_auc})

        print(
            f'\tTraining step: Loss: {train_loss}, AUC: {train_auc}', end='\n')

        val_loss, val_auc = test(model=model, data_loader=validation_dataset_loader,
                                 loss_function=loss, device=device)
        validation_updater.update(**{"loss": val_loss, "accuracy": val_auc})

        print(f'\tValidation step: Loss: {val_loss}, AUC: {val_auc}', end='\n')

        torch.save(model.state_dict(), f'checkpoint_epoch_{t}.pth')
        print('\tModel saved')

        model_checkpoint.update(loss=val_loss)
        scheduler.step(val_loss)

    train_updater.display()
    validation_updater.display()


def get_prediction(model, test_dataset_loader, device, model_name):
    with torch.no_grad():
        model.eval()
        probs = []
        size = len(test_dataset_loader)
        i = 0
        for data in test_dataset_loader:
            print(f"Predicted {int(i*100/size)}% of the data", end="\r")
            if model_name == "efficientnet":
                image_input = data["image_efficientnet"].to(device)
                output = model(image_input)
            elif model_name == "resnet":
                image_input = data["image_resnet"].to(device)
                output = model(image_input)
            else:
                image_260 = data["image_efficientnet"].to(device)
                image_256 = data["image_resnet"].to(device)
                output = model(image_260=image_260, image_256=image_256)

            probs.append(output)
            i += 1
        probs = np.concatenate(probs)

    return probs


def main_predict(cfg, model_name, verbose=False):
    try:
        model_path = cfg["models"][model_name]
        print(model_path)
        if model_name == "resnet":
            model = Resnet(cfg=cfg)
        elif model_name == "efficientnet":
            model = EfficientNet(cfg=cfg)
        else:
            resnet = Resnet(cfg=cfg)
            efficientnet = EfficientNet(cfg=cfg)
            model = Ensemble(resnet=resnet, efficientnet=efficientnet)
    except:
        print("Wrong model name")
        return

    device = get_device()
    model.to(device)
    model.eval()
    df_test = pd.read_csv(f'{cfg["DATASET_DIR"]}/sample_submission.csv')

    test_dataset = DatasetTransformer_prediction(
        cfg=cfg, df=df_test)
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True)

    if verbose:
        summary(model, (3, cfg["IMAGE_SIZE"], cfg["IMAGE_SIZE"]))

    predictions = get_prediction(
        model, test_dataset_loader, device, model_name)

    df_test[cfg["TARGET_COLS"]] = predictions
    df_test[[cfg["IMAGE_COL"]] + cfg["TARGET_COLS"]
            ].to_csv('submission.csv', index=False)

    if verbose:
        print(df_test.head(20))


class GraphUpdater():
    def __init__(self, type, name=None):
        self.type = type
        if name is None:
            name = self.type + "_loss_auc_" + \
                str(int(datetime.timestamp(datetime.now())))
        self.name = name
        self.loss = []
        self.accuracy = []
        self.epoch = []

    def update(self, loss, accuracy):
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.epoch.append(len(self.epoch))
        df = pd.DataFrame(
            data={"loss": self.loss, "accuracy": self.accuracy, "epoch": self.epoch})
        df.to_csv(self.name + ".csv", index_label="epoch")

    def display(self):
        df = pd.DataFrame(
            data={"loss": self.loss, "accuracy": self.accuracy, "epoch": self.epoch})
        df.plot(x="epoch", y=["loss", "accuracy"], title="[" +
                self.type + "]" + " Loss and Accuracy per epoch")
        plt.show()
        plt.savefig(self.name)


if __name__ == "__main__":
    pass
