import pandas as pd
from sklearn.datasets import make_circles  # from sklearn Toy datasets
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import requests
from pathlib import Path

# Introducing the data. Toy dataset - small enough to experiment and still sizeable for practice
n_samples = 1000

X, y = make_circles(n_samples, noise=0.09, random_state=42)
circles = pd.DataFrame({'X1': X[:, 0],
                        'X2': X[:, 1],
                        'label': y})

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# torch.manual_seed will not affect scikit-learn code

device = "cuda" if torch.cuda.is_available() else "cpu"  # like a good manner


class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # self.layer_1 = nn.Linear(in_features=2, out_features=5)  # 2 features to 5, better use powers of 8
        # self.layer_2 = nn.Linear(in_features=5, out_features=1)  # output layer, lining up with previous one

        self.two_linear = nn.Sequential(
            nn.Linear(in_features=2, out_features=5),
            nn.Linear(in_features=5, out_features=1)
        )

    def forward(self, x: torch.Tensor):
        return self.two_linear(x)


model_0 = CircleModelV0().to(device)
with torch.inference_mode():
    unt_preds = model_0(X_test.to(device))
# print(len(unt_preds), unt_preds.shape)
# print(len(X_test), X_test.shape)
# print(f'\n {unt_preds[:10]}')
# print(y_test[:10])

l_fn = nn.BCELoss()  # inputs should go through sigmoid before it
loss_fn = nn.BCEWithLogitsLoss()  # sigmoid activation function build in == BCE loss + sigmoid layer

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)


# Calculate accuracy = TP / (TP + TN) * 100 (not the best metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # for pytorch it works
    acc = (correct / len(y_pred)) * 100
    return acc


model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]

# Sigmoid act. function to receive prediction probabilities

y_pred_prob = torch.sigmoid(y_logits)

# Find the predicted labels
y_preds = torch.round((y_pred_prob))

# In full (logits -> pred probs -> pred labels)

y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))
# print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))
# print(y_preds.squeeze())

# BUILDING LOOPS

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 100

# putting data to the target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_0.train()

    # Forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # Loss and accuracy

    loss = loss_fn(y_logits, y_train)  # BCEWithLogitsLoss not just BCE
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Loss backward
    loss.backward()

    # Optimizer step

    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)

    # Printing

    if epoch % 100 == 0:
        print(f'Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}%, | Test loss: {test_loss:.5f}'
              f', Test acc: {test_acc:.2f}%')

# Predictions

def pirating():
    if Path('helper_functions.py').is_file():
        print("helper_functions.py already exists, skipping download")
    else:
        print("Downloading helper_functions.py")
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
        with open("helper_functions.py", "wb") as f:  # in binary state
            f.write(request.content)  # for binary response content

# pirating()

from helper_functions import plot_predictions, plot_decision_boundary

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_0, X_train, y_train)
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(model_0, X_test, y_test)
# plt.show()

## IMPROVING MODEL

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(in_features=2, out_features=10)
        self.l2 = nn.Linear(in_features=10, out_features=10)
        self.l3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.l3(self.l2(self.l1(x)))


model_1 = CircleModelV1().to(device)
# print(model_1)

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# for epoch in range(epochs):
#     model_1.train()
#
#     y_logits = model_1(X_train).squeeze()
#     y_pred = torch.round(torch.sigmoid(y_logits))
#
#     loss = loss_fn(y_logits, y_train)
#     acc = accuracy_fn(y_train, y_pred)
#
#     optimizer.zero_grad()
#
#     loss.backward()
#
#     optimizer.step()
#
#     model_1.eval()
#     with torch.inference_mode():
#         test_logits = model_1(X_test).squeeze()
#         test_pred = torch.round(torch.sigmoid(test_logits))
#
#         test_loss = loss_fn(test_logits, test_pred)
#         test_acc = accuracy_fn(y_test, test_pred)
#
#     # Printing
#
#     if epoch % 100 == 0:
#         print(f'Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}%, | Test loss: {test_loss:.5f}'
#               f', Test acc: {test_acc:.2f}%')



# Something is missing - non-linearity

n_samples = 1000

X, y = make_circles(n_samples, noise=0.03, random_state=42)
circles = pd.DataFrame({'X1': X[:, 0],
                        'X2': X[:, 1],
                        'label': y})

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(in_features=2, out_features=10)
        self.l2 = nn.Linear(in_features=10, out_features=10)
        self.l3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()  # non-linear activation function

    def forward(self, x: torch.Tensor):
        return self.l3(self.relu(self.l2(self.relu(self.l1(x)))))


model_2 = CircleModelV2().to(device)


loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

torch.manual_seed(42)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

epochs = 1300

for epoch in range(epochs):
    model_2.train()

    y_logits = model_2(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_2.eval()
    with torch.inference_mode():
        test_logits = model_2(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)

    if epoch % 100 == 0:
        print(f'Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}%, | Test loss: {test_loss:.5f}'
               f', Test acc: {test_acc:.2f}%')


model_2.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_2(X_test))).squeeze()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Train')
plot_decision_boundary(model_2, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_2, X_test, y_test)
plt.show()