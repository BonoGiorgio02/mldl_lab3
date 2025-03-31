import torch
from torch import nn

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2)   # B x 64 x 112 x 112
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2) # B x 128 x 56 x 56
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)# B x 256 x 28 x 28
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)# B x 512 x 14 x 14
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2)# B x 1024 x 7 x 7

        # Fully Connected layers
        self.flatten = nn.Flatten(1)  # Modificato da 2 a 1
        self.fc1 = nn.Linear(1024 * 7 * 7, 512)  # Modificato da 128*56*56 a 1024*7*7
        self.fc2 = nn.Linear(512, 200)  # Output con 200 classi

        # Funzione di attivazione
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # Convolutional layers con ReLU
        x = self.relu(self.conv1(x))  # B x 64 x 112 x 112
        x = self.relu(self.conv2(x))  # B x 128 x 56 x 56
        x = self.relu(self.conv3(x))  # B x 256 x 28 x 28
        x = self.relu(self.conv4(x))  # B x 512 x 14 x 14
        x = self.relu(self.conv5(x))  # B x 1024 x 7 x 7

        # Flatten per passare alla parte fully connected
        x = self.flatten(x)           # B x (1024*7*7)

        # Fully connected layers con ReLU
        x = self.relu(self.fc1(x))    # B x 512
        x = self.dropout(x)
        x = self.fc2(x)               # B x 200 (output finale con softmax da applicare dopo)

        return x
