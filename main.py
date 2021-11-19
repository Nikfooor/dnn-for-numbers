import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch.nn.functional as F
import torch.cuda
import matplotlib.pyplot as plt
import random


class MnistModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes, device):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, num_classes)
        self.device = device

    def forward(self, imag):
        imag = imag.reshape(-1, 784)  # меняем форму с 3d [100, 28, 28] на 2d [100, 784]
        out = self.linear1(imag)  # делаем предсказание, получаем массив [100, 10] 784->10 784 нейрона на вход и 10 на выход
        out = F.relu(out)  # применяем функцию активации
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)  # вычисляем последний слой
        return out  # возвращаем его

    def training_step(self, batch):
        images, labels = batch[0].to(self.device), batch[1].to(self.device)
        out = self(images)  # обращаемся к out, получаем предсказание
        loss = F.cross_entropy(out, labels)  # вычисляем степень loss, то есть энтропию, чем она больше, тем ровнее распределены вероятности
        return loss  # возращаем значени loss

    def validation_step(self, batch):
        images, labels = batch[0].to(self.device), batch[1].to(self.device)
        out = self(images)  # обращаемся к out, получаем предсказание
        loss = F.cross_entropy(out, labels)  # вычисляем степень loss
        acc = accuracy(out, labels)  # вычисляем точность между предсказаниями и целью
        return {'val_loss': loss, 'val_acc': acc}  # возвращаем словарь

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]  # список losses
        epoch_loss = torch.stack(batch_losses).mean()  # вычисляет среднее
        batch_accs = [x['val_acc'] for x in outputs]  # список accs
        epoch_acc = torch.stack(batch_accs).mean()    # вычисляет среднее
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}  # возвращает средние значение

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


def accuracy(outputs, labels):  # вычисление точности наших предсказаний
    _, preds = torch.max(outputs, dim=1)  # выбираем наибольшую "вероятность", которой соотсветсвует число от 0 до 9
    return torch.tensor(torch.sum(preds == labels).item()/len(labels))  # сравниваем предсказанные числа с известными числами и считаем отношение кол-ва правильных предсказаний к общему кол-ву


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]  # вычисляем loss и acc и записываем их в список словарей
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)  # функция улучшения значения весов
    history = []

    for epoch in range(epochs):
        # Training
        for batch in train_loader:
            loss = model.training_step(batch)  # вычислили loss
            loss.backward()  # подготавливаем производную(градиент)
            optimizer.step()  # делаем шаг(вычисляя производную) и улучшаем значения весов
            optimizer.zero_grad()  # обнуляем производную

        # обучение нейросети закончено, дальше идет проверка точности

        # Validation
        result = evaluate(model, val_loader)  # вычисляем средние значения loss и accs
        model.epoch_end(epoch, result)  # печатаем их
        history.append(result)  # добавляем в массив результаты

    return history  # возвращаем массив результатов


def predict_number(img, model, device):
    xb = img.unsqueeze(0).to(device)
    yb = model(xb).to(device)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()


def main():

    # TRAINING PARAMETERS
    batch_size = 100
    input_size = 28*28
    hidden_size1 = 512
    hidden_size2 = 512
    hidden_size3 = 512
    num_classes = 10
    training_speed = 0.5
    device = 'cpu'

    if torch.cuda.is_available():
        device = input("Введите девайс (cpu/cuda): ")
    epochs_num = int(input("Введите количество эпох:"))

    test_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor())
    dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())
    train_ds, val_ds = random_split(dataset, [50000, 10000])
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, num_workers=1, pin_memory=True)

    img, label = test_dataset[random.randint(1, 10000)]

    model = MnistModel(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes, device)
    model.to(device)

    t1 = time.time()
    history = fit(epochs_num, training_speed, model, train_loader, val_loader)
    t2 = time.time()
    print("Time for execution: {:.4f} seconds".format(t2-t1))

    print(predict_number(img, model, device))
    plt.imshow(img[0], cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
