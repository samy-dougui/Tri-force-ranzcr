import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_data(mode="dev", percentage=0.5):
    data = pd.read_csv(
        '/kaggle/input/ranzcr-clip-catheter-line-classification/train.csv')
    if mode == "prod":
        return data
    else:
        print(f"Dev mode used, percentage of the data used: {percentage}")
        return data.sample(frac=percentage)


class DatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        if transform is None:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((image_size, image_size))])
        else:
            self.transform = transform
        self.df = df
        self.labels = self.df[target_cols].values
        self.file_names = df['StudyInstanceUID'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.file_names[idx]
        image_path = f'../input/ranzcr-clip-catheter-line-classification/train/{image_name}.jpg'
        image = Image.open(image_path)
        image_t = self.transform(image)
        label = self.labels[idx]
        data = {
            "image": image_t,
            target_cols[0]: torch.tensor(label[0], dtype=torch.float32).view(1),
            target_cols[1]: torch.tensor(label[1], dtype=torch.float32).view(1),
            target_cols[2]: torch.tensor(label[2], dtype=torch.float32).view(1),
            target_cols[3]: torch.tensor(label[3], dtype=torch.float32).view(1),
            target_cols[4]: torch.tensor(label[4], dtype=torch.float32).view(1),
            target_cols[5]: torch.tensor(label[5], dtype=torch.float32).view(1),
            target_cols[6]: torch.tensor(label[6], dtype=torch.float32).view(1),
            target_cols[7]: torch.tensor(label[7], dtype=torch.float32).view(1),
            target_cols[8]: torch.tensor(label[8], dtype=torch.float32).view(1),
            target_cols[9]: torch.tensor(label[9], dtype=torch.float32).view(1),
            target_cols[10]: torch.tensor(
                label[10], dtype=torch.float32).view(1)
        }

        return data


def linear_relu(dim_in, dim_out):
    return [nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True)]


class VanillaModel(nn.Module):
    def __init__(self, input_size):
        super(VanillaModel, self).__init__()
        self.input_size = input_size
        self.classifier = nn.Sequential(
            *linear_relu(self.input_size, self.input_size),
            *linear_relu(self.input_size, 2*self.input_size),
        )

        self.out1 = nn.Linear(2*self.input_size, 1)
        self.out2 = nn.Linear(2*self.input_size, 1)
        self.out3 = nn.Linear(2*self.input_size, 1)
        self.out4 = nn.Linear(2*self.input_size, 1)
        self.out5 = nn.Linear(2*self.input_size, 1)
        self.out6 = nn.Linear(2*self.input_size, 1)
        self.out7 = nn.Linear(2*self.input_size, 1)
        self.out8 = nn.Linear(2*self.input_size, 1)
        self.out9 = nn.Linear(2*self.input_size, 1)
        self.out10 = nn.Linear(2*self.input_size, 1)
        self.out11 = nn.Linear(2*self.input_size, 1)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)

        out1 = torch.sigmoid(self.out1(x))
        out2 = torch.sigmoid(self.out2(x))
        out3 = torch.sigmoid(self.out3(x))
        out4 = torch.sigmoid(self.out4(x))
        out5 = torch.sigmoid(self.out5(x))
        out6 = torch.sigmoid(self.out6(x))
        out7 = torch.sigmoid(self.out7(x))
        out8 = torch.sigmoid(self.out8(x))
        out9 = torch.sigmoid(self.out9(x))
        out10 = torch.sigmoid(self.out10(x))
        out11 = torch.sigmoid(self.out11(x))

        return out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11


def BCE_loss(outputs, targets):
    o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11 = outputs
    t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11 = targets

    l1 = nn.BCELoss()(o1, t1)
    l2 = nn.BCELoss()(o2, t2)
    l3 = nn.BCELoss()(o3, t3)
    l4 = nn.BCELoss()(o4, t4)
    l5 = nn.BCELoss()(o5, t5)
    l6 = nn.BCELoss()(o6, t6)
    l7 = nn.BCELoss()(o7, t7)
    l8 = nn.BCELoss()(o8, t8)
    l9 = nn.BCELoss()(o9, t9)
    l10 = nn.BCELoss()(o10, t10)
    l11 = nn.BCELoss()(o11, t11)

    return (l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9 + l10 + l11)/11


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
    model.train()
    for _, data in enumerate(data_loader):
        #print(f"[TRAIN] batch number: {i}")
        image_input = data["image"].to(device)
        target1 = data["ETT - Abnormal"].to(device)
        target2 = data["ETT - Borderline"].to(device)
        target3 = data["ETT - Normal"].to(device)
        target4 = data["NGT - Abnormal"].to(device)
        target5 = data["NGT - Borderline"].to(device)
        target6 = data["NGT - Incompletely Imaged"].to(device)
        target7 = data["NGT - Normal"].to(device)
        target8 = data["CVC - Abnormal"].to(device)
        target9 = data["CVC - Borderline"].to(device)
        target10 = data["CVC - Normal"].to(device)
        target11 = data["Swan Ganz Catheter Present"].to(device)

        targets = (target1, target2, target3, target4, target5,
                   target6, target7, target8, target9, target10, target11)
        outputs = model(image_input)

        loss = loss_function(outputs, targets)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


def compute_correctly_predicted(outputs, targets, eps=0.3):
    correctly_predicted = 0.0
    for output, target in zip(outputs, targets):
        correctly_predicted += (abs(output-target) < eps).sum().item()
    return correctly_predicted


def test(model, data_loader, loss_function, device):
    with torch.no_grad():
        model.eval()
        total_number = 0
        total_loss, correctly_predicted = 0.0, 0.0

        for _, data in enumerate(data_loader):
            image_input = data["image"].to(device)
            target1 = data["ETT - Abnormal"].to(device)
            target2 = data["ETT - Borderline"].to(device)
            target3 = data["ETT - Normal"].to(device)
            target4 = data["NGT - Abnormal"].to(device)
            target5 = data["NGT - Borderline"].to(device)
            target6 = data["NGT - Incompletely Imaged"].to(device)
            target7 = data["NGT - Normal"].to(device)
            target8 = data["CVC - Abnormal"].to(device)
            target9 = data["CVC - Borderline"].to(device)
            target10 = data["CVC - Normal"].to(device)
            target11 = data["Swan Ganz Catheter Present"].to(device)

            targets = (target1, target2, target3, target4, target5,
                       target6, target7, target8, target9, target10, target11)
            outputs = model(image_input)

            total_number += image_input.shape[0]

            # The multiplication by inputs.shape[0] is due to the fact
            # that our loss criterion is averaging over its samples
            total_loss += image_input.shape[0] * \
                loss_function(outputs, targets).item()
            correctly_predicted += compute_correctly_predicted(
                outputs, targets)
    return total_loss / total_number, correctly_predicted / total_number


if __name__ == "__main__":
    batch_size = 128
    epochs = 10
    device = get_device()

    image_size = 200
    target_size = 11
    target_cols = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                   'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
                   'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                   'Swan Ganz Catheter Present']
    data = get_data(mode="dev")

    train_validation_frac = 0.8
    train_data = data.sample(frac=train_validation_frac)
    validation_data = data.drop(train_data.index)

    train_dataset = DatasetTransformer(train_data)
    validation_dataset = DatasetTransformer(validation_data)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_dataset_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    first_batch = next(iter(train_dataset_loader))["image"][0]
    input_size = torch.numel(first_batch[0])
    model = VanillaModel(input_size=input_size)
    model.to(device)

    learning_rate = 0.01
    weight_decay = 0.0

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=20, gamma=0.5)

    for t in range(epochs):
        print(f'Epoch: {t}')
        scheduler.step()

        train(model=model, data_loader=train_dataset_loader,
              loss_function=BCE_loss, optimizer=optimizer, device=device)
        val_loss, val_acc = test(model=model, data_loader=validation_dataset_loader,
                                 loss_function=BCE_loss, device=device)
        print(
            f'Validation step: Loss: {val_loss}, Accuracy: {val_acc}', end='\n')
