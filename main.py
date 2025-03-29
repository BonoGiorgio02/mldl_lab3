from models.customnet import CustomNet
from train import train
from eval import validate

import os
import shutil
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch
from torch import nn

def main():
    model = CustomNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_acc = 0

    # Run the training process for {num_epochs} epochs
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        val_accuracy = validate(model, val_loader, criterion)

        # Best validation accuracy
        best_acc = max(best_acc, val_accuracy)


    print(f'Best validation accuracy: {best_acc:.2f}%')

    
if __name__ == "__main__":
    main()