from models.customnet import CustomNet
import torch
from torch import nn
from utils.download_dataset import val_loader

def validate(model, val_loader, criterion):
    model.eval()  # Imposta il modello in modalità di valutazione (disabilita dropout, batchnorm, ecc.)
    val_loss = 0  # Variabile per accumulare la perdita totale durante la validazione

    correct, total = 0, 0  # Variabili per calcolare l'accuratezza

    with torch.no_grad():  # Disabilita il calcolo dei gradienti (più veloce e meno memoria)
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()  # Sposta i dati su GPU

            # Passa i dati attraverso il modello
            outputs = model(inputs)

            # Calcola la perdita
            loss = criterion(outputs, targets)

            val_loss += loss.item()  # Aggiungi la perdita di questo batch all'accumulatore

            # Calcola le previsioni (indice della classe con probabilità massima)
            _, predicted = outputs.max(1)  # max(1) restituisce il valore massimo e l'indice della classe

            total += targets.size(0)  # Incrementa il numero totale di esempi
            correct += predicted.eq(targets).sum().item()  # Conta le predizioni corrette

    # Calcola la perdita media
    val_loss = val_loss / len(val_loader)
    # Calcola l'accuratezza
    val_accuracy = 100. * correct / total

    # Stampa i risultati
    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')

    return val_accuracy

model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_acc = 0

# Run the training process for {num_epochs} epochs
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    # At the end of each training iteration, perform a validation step
    val_accuracy = validate(model, val_loader, criterion)
    # Best validation accuracy
    best_acc = max(best_acc, val_accuracy)

print(f'Best validation accuracy: {best_acc:.2f}%')
