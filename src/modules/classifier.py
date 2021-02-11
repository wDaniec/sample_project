import torch
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lightning import LightningModule


class Classifier(LightningModule):
    def __init__(self, model, lr=0.1):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        return loss_val

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return optimizer

