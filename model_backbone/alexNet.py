import torch
import torch.nn as nn

class AlexNet(nn.Module):   
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 64x64x3 -> 32x32x64
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # 32x32x64 -> 16x16x64
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 16x16x64 -> 16x16x192
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # 16x16x192 -> 8x8x192
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 8x8x192 -> 8x8x384
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 8x8x384 -> 8x8x256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 8x8x256 -> 8x8x256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 8x8x256 -> 4x4x256
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.classifier = nn.Sequential(
            # 4x4x256 = 4096
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.fc = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.fc(x)
        return x
        
