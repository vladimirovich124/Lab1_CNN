import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Определяем сверточные слои
        self.conv_layers = nn.Sequential(
            self._conv_block(3, 32),    # Первый сверточный блок
            self._conv_block(32, 64),   # Второй сверточный блок
            self._conv_block(64, 128)   # Третий сверточный блок
        )
        # Полносвязные слои
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512), # Полносвязный слой
            nn.ReLU(),                   # Активация ReLU
            nn.Linear(512, 10)           # Выходной слой
        )

    # Вспомогательная функция для создания сверточного блока
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv_layers(x)           # Проходим через сверточные слои
        x = x.view(-1, 128 * 4 * 4)       # Разворачиваем тензор перед полносвязным слоем
        x = self.fc_layers(x)             # Проходим через полносвязные слои
        return x