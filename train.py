from models.customnet import CustomNet

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # Calcola le previsioni del modello
        outputs = model(inputs)

        # Calcola la perdita
        loss = criterion(outputs, targets)

        # Azzera i gradienti
        optimizer.zero_grad()

        # Calcola i gradienti attraverso il backward pass
        loss.backward()

        # Aggiorna i pesi
        optimizer.step()


        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')