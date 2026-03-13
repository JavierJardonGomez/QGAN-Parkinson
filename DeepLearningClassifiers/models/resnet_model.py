import torch.nn as nn
import torch.nn.functional as F


# Modelo
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(8, 5, 3), dropout=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_sizes[0], padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_sizes[1], padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_sizes[2], padding='same')
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.dropout(out)
        
        out += residual
        return F.relu(out)

class Conv1DResidualClassifier(nn.Module):
    def __init__(self, input_channels, nb_classes):
        super(Conv1DResidualClassifier, self).__init__()
        n_feature_maps = 64

        self.block1 = ResidualBlock(input_channels, n_feature_maps)
        self.block2 = ResidualBlock(n_feature_maps, n_feature_maps * 2)
        self.block3 = ResidualBlock(n_feature_maps * 2, n_feature_maps * 2, dropout=True)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Output shape (batch, channels, 1)
        self.fc = nn.Linear(n_feature_maps * 2, nb_classes)

    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        out = self.global_avg_pool(out).squeeze(-1)  # Shape: (batch_size, channels)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)