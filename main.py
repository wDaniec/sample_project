from src.models import ResNet18
from src.data import half_cifar, cifar
from src.modules import Classifier
import pytorch_lightning as pl


def main():
    model = ResNet18(100)
    train_loader, valid_loader = half_cifar(variant="100")
    print(train_loader, valid_loader)
    pl_module = Classifier(model)
    trainer = pl.Trainer(gpus=1, max_epochs=20)
    trainer.fit(pl_module, train_loader, valid_loader)

if __name__ == "__main__":
    main()
