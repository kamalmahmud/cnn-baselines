import torch.nn as nn


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
    

# Best accuracy achieved: 99.28%, trained on a P100 GPU.

class MnistCNN_V1(nn.Module):
    def __init__(self):
        super(MnistCNN_V1,self).__init__()
        # input is 1x28x28, so input channels = 1

        self.conv1 = nn.Conv2d(1,32,kernel_size=3,padding='same')
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding='same')
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding='valid')

        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptivepool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(128,10)
        self.activation = nn.functional.relu

    def forward(self,x):
        x = self.conv1(x) # [B, 32, 28, 28]
        x = self.activation(x)
        x = self.pool(x) # [B, 32, 14, 14]   28/2 = 14


        x = self.conv2(x) # [B, 64, 14, 14]
        x = self.activation(x)
        x = self.pool(x) # [B, 64, 7, 7]


        x = self.conv3(x) # [B, 128, 5, 5]
        x = self.activation(x)
        x = self.adaptivepool(x) # [B, 128, 1, 1]

        x = x.view(x.size(0),-1) #flatten the tensor [B,128]
        x = self.fc1(x)

        return x
    

class MnistCNN_V2(nn.Module):
    def __init__(self):
        super(MnistCNN_V2,self).__init__()
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
    
