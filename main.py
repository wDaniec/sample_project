from src.models import ResNet18, SimpleCNN
from src.data import half_cifar, cifar
from src.modules import Classifier
from pytorch_lightning.loggers import NeptuneLogger
import pytorch_lightning as pl
from src import NEPTUNE_TOKEN, NEPTUNE_USER, NEPTUNE_PROJECT

def train_phase(pl_module, loaders):
    logger = get_logger("sample_experiment")
    trainer = pl.Trainer(gpus=1, max_epochs=10, log_every_n_steps=25, logger=logger)
    trainer.fit(pl_module, loaders[0], loaders[1])

def get_logger(experiment_name):
    logger = NeptuneLogger(
                api_key=NEPTUNE_TOKEN,
                project_name=NEPTUNE_USER + "/" + NEPTUNE_PROJECT,
                experiment_name= experiment_name
        )
    return logger
    

def main():
    model = SimpleCNN(100)
    half_loaders = half_cifar(variant="100")
    full_loaders = cifar(variant="100")
    pl_module = Classifier(model)
    train_phase(pl_module, half_loaders)
    train_phase(pl_module, full_loaders)

if __name__ == "__main__":
    main()
