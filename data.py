import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from collections import namedtuple

# Функция для получения меток классов
def get_class_labels():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Функция для получения преобразований данных
def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация изображений
    ])

# Функция для разделения набора данных на тренировочный и валидационный
def split_dataset(dataset, train_val_split):
    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])

# Основная функция для получения загрузчиков данных
def get_data_loaders(batch_size, train_val_split=0.8):
    transform = get_transforms()

    # Загрузка тренировочного набора данных CIFAR-10
    full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dataset, val_dataset = split_dataset(full_dataset, train_val_split)

    # Загрузка тестового набора данных CIFAR-10
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Создание загрузчиков данных
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Возвращаем загрузчики данных с использованием именованного кортежа
    return train_loader, val_loader, test_loader
