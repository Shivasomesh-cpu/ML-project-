import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3D(nn.Module):
    """
    Custom 4-layer 3D Convolutional Neural Network baseline.
    Used as an untrained (randomly initialized) baseline representation
    to quantify downstream performance of random projections vs foundation models.
    """
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(2)
        self.drop1 = nn.Dropout3d(dropout_rate)

        self.conv2 = nn.Conv3d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)
        self.drop2 = nn.Dropout3d(dropout_rate)

        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(2)
        self.drop3 = nn.Dropout3d(dropout_rate)

        self.conv4 = nn.Conv3d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(2)
        self.drop4 = nn.Dropout3d(dropout_rate)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(256, 512)
        self.drop5 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.drop1(self.bn1(self.pool1(F.relu(self.conv1(x)))))
        x = self.drop2(self.bn2(self.pool2(F.relu(self.conv2(x)))))
        x = self.drop3(self.bn3(self.pool3(F.relu(self.conv3(x)))))
        x = self.drop4(self.bn4(self.pool4(F.relu(self.conv4(x)))))
        x = self.global_pool(x).view(x.size(0), -1)
        return self.drop5(F.relu(self.fc1(x)))
