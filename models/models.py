import timm
import torch
import torch.nn as nn


class EfficientNet(nn.Module):
    def __init__(self, cfg, model_name="efficientnet_b2", pretrained=False):
        super(EfficientNet, self).__init__()

        model = timm.create_model(model_name, pretrained=pretrained)

        n_features = model.classifier.in_features

        self.model = nn.Sequential(*list(model.children())[:-1])
        self.drop_out = nn.Dropout(p=0.5)
        self.fc = nn.Linear(n_features, cfg["TARGET_SIZE"])

    def forward(self, x):
        x = self.model(x)
        x = self.drop_out(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


class Resnet(nn.Module):
    def __init__(self, cfg, model_name="resnet200d", pretrained=False):
        super(Resnet, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, cfg["TARGET_SIZE"])

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        output = torch.sigmoid(output)
        return output


class Ensemble():
    def __init__(self, resnet, efficientnet):
        self.efficientnet = efficientnet
        self.resnet = resnet

    def __call__(self, image_260, image_256):
        return (self.efficientnet(image_260) + self.resnet(image_256))/2

    def eval(self):
        self.resnet.eval()
        self.efficientnet.eval()
