import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        # input is 1x28x28, so input channels = 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')

        # we end with 64 channels. We pool twice so 28 => 14 => 7 for dimensions
        self.last_conv_length = 64 * 7 * 7
        self.fc1 = nn.Linear(self.last_conv_length, 128)
        self.fc2 = nn.Linear(128, 10)
        self.activation = nn.functional.relu

    def forward(self, x):
        # input x: [B, 1, 28, 28]
        # first Conv block
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)

        # Second Conv block
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)

        # Reshape Linearly the last layer

        x = x.view(-1, self.last_conv_length)
        x = self.fc1(x)
        x = self.activation(x)

        # don't add activation after it for classification
        x = self.fc2(x)

        return x


def get_data_loaders(path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root=path, train=True,
                                          download=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testSet = torchvision.datasets.MNIST(root=path, train=False,
                                         download=True, transform=transform)

    testLoader = torch.utils.data.DataLoader(testSet, batch_size=64, shuffle=False)

    return trainLoader, testLoader


if __name__ == '__main__':
    root = './data'
    trainLoader, testLoader = get_data_loaders(root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MnistCNN().to(device)

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