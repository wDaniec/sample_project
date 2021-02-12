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
        loss = F.cross_entropy(output, target)
        acc = self._accuracy(output, target)
        self.log("train_loss", loss, on_step=True)
        self.log("train_acc", acc, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss = F.cross_entropy(output, target)
        acc = self._accuracy(output, target)
        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def _accuracy(self, output, target):
        # print(output.shape, target.shape)
        pred = torch.argmax(output, dim=1)
        batch_size = target.shape[0]
        acc = 100 * torch.sum(pred == target).float() / batch_size
        return acc
