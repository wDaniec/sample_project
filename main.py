from src.models import SimpleCNN
from src.data import fmnist

def main():
    model = SimpleCNN(5, 32)
    train_loader, valid_loader = fmnist()
    print(model)
    print(train_loader, valid_loader)

if __name__ == "__main__":
    main()