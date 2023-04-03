import torch
from torch import nn  # nn - blocks for nn
import matplotlib.pyplot as plt
import numpy as np


# DATA PREPARING AND LOADING

weight = 0.7
bias = 0.3  # смещение (параметр тета0 в уравнении регрессии)


start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias  # simple regression

# print(X[:10], y[:10], len(X), len(y), sep='\n')

# SPLITTING DATA

train = int(0.8 * len(X))
X_train, y_train = X[:train], y[:train]
X_test, y_test = X[train:], y[train:]

def plot_pred(train_data = X_train, train_labels=y_train,test_data=X_test,test_labels=y_test,
              predictions=None):
    """
    Plots training data, test data and compares predictions
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label='Training data')

    plt.scatter(test_data,test_labels, c='g', s=4, label='Testing data')

    if predictions is not None:
        plt.scatter(test_data,predictions, c='r', s=4, label='Predictions')

    plt.legend(prop={"size": 14})
    plt.show()


# SOME CLASS LEARNING STUFF, NOT FROM VIDEO
class Dog:
    species = "Canis familiaris"

    def __init__(self, name, age):
        self.name = name
        self.age = age

    # Instance method
    def __str__(self):
        return f"{self.name} is {self.age} years old"

    # Another instance method
    def speak(self, sound):
        return f"{self.name} says {sound}"
class JackRusselTerrier(Dog):
    def speak(self, sound="Arf"):
        return super().speak(sound)
class GoldenRetriever(Dog):
    def speak(self, sound='Bark'):
        return super().speak(sound)
miles = JackRusselTerrier("Miles", 4)
class Car:
    def __init__(self,color, mileage):
        self.color = color
        self.mileage = mileage

    def __str__(self):
        return f'The {self.color} car has {self.mileage} miles.'
# blue = Car('blue', 20_000)
# red = Car(color='red', mileage=30_000)
#
# for car in (blue, red):
#     print(car)

# Create LR class

class LinRegModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))  # случайные
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))  # начальные числа, которые и будут дальше подбираться

    def forward(self, x: torch.Tensor):
        return self.weights * x + self.bias


torch.manual_seed(42)

model_0 = LinRegModel()
# print(list(model_0.parameters()))  # параметры
# print(model_0.state_dict())

with torch.inference_mode():  # отключает отслеживание градиентов (меньше данных тречится в памяти, т.е. быстрее)
    y_preds = model_0(X_test)  #  inference_mode == no_grad

# Задаём функцию потерь
loss_fn = nn.L1Loss()  # абсолютная ошибка (mean of abs (y_true - y_pred)) (MAE - mean absolute error)

# Optimizer (stochastic (random) gradient descent)
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)  # подбирает параметры, чтобы уменьшить функцию потерь (аля метод подбора?)
# lr = learning rate = коэф скорости обучения (шаг)


# epoch - one loop through the data
epochs = 200

epoch_count = []
loss_values = []
test_loss_values = []


# Loop through the data
for epoch in range(1, epochs + 1):
    # Set the model to training mode
    model_0.train()  # sets all parameters that require grad require grad

    # 1. Forward pass
    y_pred = model_0(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()  # == count = 0 each iteration

    # 4. Perform backpropagation
    loss.backward()

    # 5. Step the optimizer

    optimizer.step()  # it accumulates through the loop, so that is why 3. exist

    ### Testing
    model_0.eval()
    with torch.inference_mode():  # turn off gradient tracking
        test_pred = model_0(X_test)

        # calculate loss

        test_loss = loss_fn(test_pred, y_test)

    epoch_count.append(epoch)
    loss_values.append(loss)
    test_loss_values.append(test_loss)

    # if epoch % 50 == 0:
    #     print(f'Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}')
    #     print(model_0.state_dict())


def plotting(loss_values = loss_values, test_loss_values = test_loss_values):
    loss_values = np.array(torch.tensor(loss_values).numpy())
    test_loss_values = np.array(torch.tensor(test_loss_values).numpy())

    plt.plot(epoch_count, loss_values, label='Train loss')
    plt.plot(epoch_count, test_loss_values, label='Test loss')
    plt.title('Training and test loss curves')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.grid()
    plt.legend()

    plot_pred(predictions=test_pred)


# SAVING A MODEL IN PYTORCH
torch.save(model_0.state_dict(), 'models.pth')  # расширение для сохранения pt или pth. Полезная библиотека Path

# LOADING MODEL
# We need to instantiate a new instance of model's class
loaded_model_0 = LinRegModel()
loaded_model_0.load_state_dict(torch.load('models.pth'))

# Making predictions

loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

# print(test_pred == loaded_model_preds)




# FINAL WORKFLOW

# setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

weight = 0.7
bias = - 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

train = int(0.8 * len(X))
X_train, y_train = X[:train], y[:train]
X_test, y_test = X[train:], y[train:]

# plot_pred(X_train,y_train, X_test, y_test)


class LinRegModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # use nn.Linear() instead of initializing parameters by hand
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor):
        return self.linear_layer(x)


torch.manual_seed(42)
model_1 = LinRegModelV2()
print(model_1.state_dict())
model_1.to(device)
# print(next(model_1.parameters()).device)

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

torch.manual_seed(42)

epochs = 200

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)


for epoch in range(epochs):
    model_1.train()

    y_pred = model_1(X_train)

    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    model_1.eval()

    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        print(f'Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}')


model_1.eval()
with torch.inference_mode():
    y_preds = model_1(X_test)

# plot_pred(predictions=y_preds.cpu())