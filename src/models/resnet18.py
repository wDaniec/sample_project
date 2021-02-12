import torch.nn as nn
import torchvision

class ResNet18(nn.Module):
    def __init__(self, out_features):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18()
        in_features = self.model.fc.in_features
        last_layer = nn.Linear(in_features, out_features)
        self.model.fc = last_layer

    def forward(self, x):
        return self.model(x)