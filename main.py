from src.models import SimpleCNN
from src.data import fmnist
from src.modules import Classifier
import pytorch_lightning as pl


def main():
    model = SimpleCNN(5, 32)
    train_loader, valid_loader = fmnist()
    pl_module = Classifier(model)
    trainer = pl.Trainer(gpus=1, max_epochs=20)
    trainer.fit(pl_module, train_loader, valid_loader)

if __name__ == "__main__":
    main()
