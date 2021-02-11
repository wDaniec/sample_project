from src.models import SimpleCNN
from src.data import fmnist
from src.modules import Classifier

def main():
    model = SimpleCNN(5, 32)
    train_loader, valid_loader = fmnist()
    pl_module = Classifier(model)
    print(model)
    print(train_loader, valid_loader)
    print(pl_module)

if __name__ == "__main__":
    main()