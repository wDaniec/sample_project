from src.models import ResNet18, SimpleCNN
from src.data import half_cifar, cifar
from src.modules import Classifier
from pytorch_lightning.loggers import NeptuneLogger
import pytorch_lightning as pl
from src import NEPTUNE_TOKEN, NEPTUNE_USER, NEPTUNE_PROJECT, USE_NEPTUNE


def get_trainer(num_epochs):
    loggers = []
    if USE_NEPTUNE:
        logger = get_logger("sample_experiment")
        loggers.append(logger)
    trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, log_every_n_steps=25, logger=loggers)
    return trainer

def get_logger(experiment_name):
    logger = NeptuneLogger(
                api_key=NEPTUNE_TOKEN,
                project_name=NEPTUNE_USER + "/" + NEPTUNE_PROJECT,
                experiment_name= experiment_name
        )
    return logger

def warm_start(model, num_epochs):
    half_loaders = half_cifar(variant="10")
    full_loaders = cifar(variant="10")
    pl_module = Classifier(model)

    trainer = get_trainer(num_epochs)
    trainer.fit(pl_module, half_loaders[0], half_loaders[1])
    trainer.fit(pl_module, full_loaders[0], full_loaders[1])

def basic_training(model, num_epochs):
    loaders = cifar(variant="10")
    pl_module = Classifier(model, lr=0.01)

    trainer = get_trainer(num_epochs)
    trainer.fit(pl_module, loaders[0], loaders[1])

def main():
    model = ResNet18(10)
    warm_start(model, num_epochs=150)
    basic_training(model, num_epochs=300)

if __name__ == "__main__":
    main()
