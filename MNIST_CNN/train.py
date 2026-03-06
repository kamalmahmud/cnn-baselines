import torch
import torch.nn as nn
import torch.optim as optim

from data import get_data_loaders
from model import MnistCNN, MnistCNN_V1, MnistCNN_V2

if __name__ == '__main__':
    root = './data'
    trainLoader, testLoader = get_data_loaders(root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MnistCNN_V2().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Model Training
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)

            optimizer.zero_grad()

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(trainLoader)}")
    print("Finished Training")

    # Model Evaluation
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy on the test set: {100.0 * correct / total:.2f}%")
