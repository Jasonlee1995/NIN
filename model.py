import torch
import torch.nn as nn

class NIN(nn.Module):
    def __init__(self, num_classes=10, init_weights=True):
        super(NIN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True),
                                   nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True),
                                   nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout1 = nn.Dropout2d()
        
        self.conv2 = nn.Sequential(nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True),
                                   nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True),
                                   nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout2 = nn.Dropout2d()
        
        self.conv3 = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
                                   nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True),
                                   nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True))
        self.pool3 = nn.AvgPool2d(kernel_size=8, stride=1)
        self.softmax = nn.Softmax(dim=-1)
        
        if init_weights: self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = torch.flatten(x, 1)
        x = self.softmax(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)