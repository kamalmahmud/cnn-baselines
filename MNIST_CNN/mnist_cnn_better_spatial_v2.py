import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Best accuracy achieved: 99.15%, trained on a P100 GPU.
# This model uses convolutional layer for classification instead of fully connected layers.

class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN,self).__init__()
        # input is 1x28x28, so input channels = 1
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,padding='same') #28
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding='same') #14
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding='valid')#5

        self.conv4 = nn.Conv2d(128,10,kernel_size=1)


        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.adaptivepool = nn.AdaptiveAvgPool2d(1)

        self.activation = nn.functional.relu


    def forward(self,x):
        x = self.conv1(x)# [B, 32, 28, 28]
        x = self.activation(x)
        x = self.pool(x)# [B, 32, 14, 14] 28/2 = 14

        x = self.conv2(x) # [B, 64, 14, 14]
        x = self.activation(x)
        x = self.pool(x)# [B, 64, 7, 7] 14/2 = 7

        x = self.conv3(x)# [B, 128, 5, 5] 7-2 due to padding
        x = self.activation(x)
        x = self.adaptivepool(x) # [B, 128, 1, 1] pool 5x5

        x = self.conv4(x) # [B, 10]           1x1 conv
        x = x.view(x.size(0),-1) # flatten x

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
