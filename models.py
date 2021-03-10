import timm
import torch
import torch.nn as nn

class EfficientNet(nn.Module):
    def __init__(self, model_name, cfg, pretrained=False):
        super(EfficientNet).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        for param in self.model.parameters():
            param.requires_grad = False
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, cfg["TARGET_SIZE"])

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x