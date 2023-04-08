import torch
import pandas as pd
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader  # it helps to load data into the model
from helper_functions import accuracy_fn
from timeit import default_timer as timer  # just for measuring diffferece between cpu and gpu time
from tqdm.auto import tqdm
import random
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sn


# We are going to build a multi-class problem on FashionMNIST Dataset. Downloading it
train_data = datasets.FashionMNIST(root='Data', train=True, download=False, transform=ToTensor(),
                                   target_transform=None)  # transform as it is in PIL format, last - about labels
test_data = datasets.FashionMNIST(root='Data', train=False, download=False, transform=ToTensor())

image, label = train_data[0]
# print(image.shape, label)
# image size is 1 x 28 x 28, where 1 = color channel, then goes h, w  #CHW order
# NCHW - N stands for batch_size; and NHWC is better in performing

classes_data = train_data.classes  # 10 classes

# VISUALIZATION
def visualdata(classes=classes_data, data=train_data, n_samples=1):
    if n_samples == 1:
        image, label = data[0]
        print(f'Image shape: {image.shape}')
        plt.imshow(image.squeeze())
        plt.title(classes[label])
        plt.show()
    if n_samples > 1:
        torch.manual_seed(42)
        fig = plt.figure(figsize=(9, 9))
        rows, cols = round(n_samples / 2), round(n_samples / 2)
        for i in range(1, rows * cols + 1):
            rand_ind = torch.randint(0, len(train_data), size=[1]).item()
            image, label = train_data[rand_ind]
            fig.add_subplot(rows, cols, i)
            plt.imshow(image.squeeze())
            plt.title(classes_data[label])
            plt.axis(False)
        plt.show()


# visualdata(n_samples=8)

"""
Because of computational efficiency it is a good idea divide data into small chunks called bathes
Usual size of a batch is 32, often powers of 2 are used too (64, 128, 256, 512)
"""

BATCH_SIZE = 32

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True) # turns data into iterable and
# shuffles it every epoch

test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

train_features_batch, train_labels_batch = next(iter(train_dataloader))
# print(f'Размеры одного batch: {train_features_batch.shape} | Размер откликов batch: {train_labels_batch.shape}')


device = "cuda" if torch.cuda.is_available() else "cpu"


'''
Baseline model - a simple model that acts as a reference in a machine learning project. 
Its main function is to contextualize the results of trained models. 
Baseline models usually lack complexity and may have little predictive power.

The scores from these algorithms provide the required point of comparison when evaluating 
all other machine learning algorithms on your problem

In this project a baseline model consists of nn.Linear()'s, so...
'''

"""
Because we're working with image data, we're going to use a different layer to start things off.
And that's the nn.Flatten() layer.
nn.Flatten() compresses the dimensions of a tensor into a single vector.

Because nn.Linear() works with what? with vectors on input
"""

# Example
flatten_model = nn.Flatten()  # all models can do forward() pass
x = train_features_batch[0]
y = flatten_model(x)
# print(y.shape) # 1 x 784

class BaselineModel(nn.Module):
    def __init__(self, inp: int, output: int, hidden_units: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=inp, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output)
        )

    def forward(self, x):
        return self.layer_stack(x)


"""
We'll need to set the following parameters:

* input_shape=784 - this is how many features you've got going in the model, in our case, 
it's one for every pixel in the target image (28 pixels high by 28 pixels wide = 784 features).
* hidden_units=10 - number of units/neurons in the hidden layer(s), 
this number could be whatever you want but to keep the model small we'll start with 10.
* output_shape=len(class_names) - since we're working with a multi-class classification problem, 
we need an output neuron per class in our dataset.
"""

baseline_model = BaselineModel(inp=784, output=len(classes_data), hidden_units=10)
baseline_model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=baseline_model.parameters(), lr=0.1)


def train_time_comparison(start, end, device=None):
    total_time = end - start
    print(f'Total time on {device}: {total_time:.3f} seconds')
    return total_time


"""
Our data batches are contained within our DataLoaders, train_dataloader and 
test_dataloader for the training and test data splits respectively.

A batch is BATCH_SIZE samples of X (features) and y (labels), 
since we're using BATCH_SIZE=32, our batches have 32 samples of images and targets.

And since we're computing on batches of data, 
our loss and evaluation metrics will be calculated per batch rather than across the whole dataset.

This means we'll have to divide our loss and accuracy values by 
the number of batches in each dataset's respective dataloader.
"""

torch.manual_seed(42)
train_cpu = timer()

epochs = 5


for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n----------")

    train_loss = 0

    for batch, (X,y) in enumerate(train_dataloader):

        X, y = X.to(device), y.to(device)

        baseline_model.train()

        y_pred = baseline_model(X)

        loss = loss_fn(y_pred, y)

        train_loss += loss  # for each batch add loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if batch % 400 == 0:
            print(f'Trained on {batch * len(X)} / {len(train_dataloader.dataset)} samples')

    train_loss /= len(train_dataloader) # mean loss per batch per epoch

    test_loss, test_acc = 0, 0
    baseline_model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:

            X, y = X.to(device), y.to(device)

            test_pred = baseline_model(X)

            test_loss += loss_fn(test_pred, y)

            test_acc += accuracy_fn(y,test_pred.argmax(dim=1))  # dim=1 required to make them same shape

        test_loss /= len(test_dataloader) # per batch per epoch
        test_acc /= len(test_dataloader)

    print(f'\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.3f}%\n')

train_cpu_end = timer()
time_baseline = train_time_comparison(train_cpu,train_cpu_end, device=str(next(baseline_model.parameters()).device))


def model_evaluation(model, data_loader, loss_fn, accuracy_fn, device=device):
    """
    :param model: A PyTorch model
    :param data_loader: target dataset for prediction
    :param loss_fn: loss function
    :param accuracy_fn: accuracy of the model

    :return: (dict) : results - name, loss, accuracy

    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y, y_pred.argmax(dim=1))

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,  # works when model created with class
            "model_loss": loss.item(),
            "model_acc": acc}


baseline_model_results = model_evaluation(baseline_model, test_dataloader, loss_fn, accuracy_fn)
# print(baseline_model_results)

# ADDING NON_LINEARITY


class ModelV1(nn.Module):

    def __init__(self, inp, out, hidden):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=inp, out_features=hidden),
            nn.ReLU(),
            nn.Linear(in_features=hidden,out_features=out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer_stack(x)


model_1 = ModelV1(inp=784,out=len(classes_data),hidden=10).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)


def training(model, data_loader, loss_fn, optimizer, accuracy_fn=accuracy_fn, device=device):
    train_loss, train_acc = 0, 0

    for batch, (X,y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)

        train_loss += loss
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


def testing(model, data_loader, loss_fn, accuracy_fn=accuracy_fn, device=device):

    test_loss, test_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for X,y in data_loader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X)

            test_loss += loss_fn(test_pred,y)
            test_acc += accuracy_fn(y, test_pred.argmax(dim=1))

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

train_gpu = timer()

epochs = 5

for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n---------')
    training(model=model_1, data_loader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer)
    testing(model=model_1, data_loader=test_dataloader, loss_fn=loss_fn)

train_gpu_end = timer()
time_model1 = train_time_comparison(start=train_gpu,end=train_gpu_end, device=device)

modelV1_results = model_evaluation(model_1, test_dataloader, loss_fn, accuracy_fn)
# Making a Convolutional NN
"""
SOURCE architecture: https://poloclub.github.io/cnn-explainer/

Main idea of CNN
Input layer -> [Convolutional layer -> activation layer -> pooling layer] -> Output layer

"""

class CNNModel(nn.Module):
    def __init__(self, inp, out, hidden):
        super().__init__()
        self.Layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=inp, out_channels=hidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # default stride value == kernel_size
        )
        self.Layer_2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden * 7 * 7, out_features=out)
        )

    def forward(self, x):
        x = self.Layer_1(x)
        # print(x.shape)
        x = self.Layer_2(x)
        # print(x.shape)  # можем узнать размер на вход в ПНН, выводя таким образом данные
        x = self.classifier(x)
        return x


model_cnn = CNNModel(inp=1, out=len(classes_data), hidden=10).to(device)
tensor = torch.randn(size=(1, 28, 28))
# print(model_cnn(tensor.unsqueeze(0).to(device)))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_cnn.parameters(), lr=0.1)

epochs = 5

train_cnn = timer()

for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n-------')
    training(model_cnn, train_dataloader, loss_fn, optimizer)

    testing(model_cnn, test_dataloader, loss_fn)

train_cnn_end = timer()

time_cnn = train_time_comparison(train_cnn, train_cnn_end)

modelCNN_results = model_evaluation(model_cnn, test_dataloader, loss_fn, accuracy_fn)

torch.save(model_cnn.state_dict(),'FashionCNN.pth')

# RESULTS COMPARISON

comp_results = pd.DataFrame([baseline_model_results, modelV1_results, modelCNN_results])
comp_results['training_time'] = [time_baseline, time_model1, time_cnn]

comp_results.set_index("model_name")["model_acc"].plot(kind="barh")
plt.grid()
plt.title(f'Сравнение работы моделей для {epochs} эпох')
plt.xlabel("accuracy (%)")
plt.ylabel("model")
plt.savefig("Comparison.png")


def make_predictions(model, data, device=device):
    pred_probs = []
    model.eval()

    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)

            pred_logit = model(sample)

            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)  # turning list into a tensor


random.seed(42)
test_sample = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_sample.append(sample)
    test_labels.append(label)

pred_probs = make_predictions(model_cnn,test_sample)
pred_classes = pred_probs.argmax(dim=1)

# Plot predictions
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_sample):

    plt.subplot(nrows, ncols, i + 1)
    plt.imshow(sample.squeeze(), cmap="gray")
    pred_label = classes_data[pred_classes[i]]
    truth_label = classes_data[test_labels[i]]
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g")  # green text if correct
    else:
        plt.title(title_text, fontsize=10, c="r")  # red text if wrong
    plt.axis(False)

# plt.show()

# MAKING A CONFUSION MATRIX

y_preds = []
model_cnn.eval()
with torch.inference_mode():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        y_logit = model_cnn(X)
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
        y_preds.append(y_pred.cpu())
y_preds = torch.cat(y_preds)

cm = confusion_matrix(test_data.targets, y_preds)
df_cm = pd.DataFrame(cm, index=[i for i in classes_data],
                     columns=[i for i in classes_data])

plt.figure(figsize=(12,7))
sn.heatmap(df_cm, annot=True, fmt="d")
plt.savefig('ConfMatrix.png')

df_cm_prob = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index=[i for i in classes_data],
                     columns=[i for i in classes_data])
plt.figure(figsize=(12, 7))
sn.heatmap(df_cm_prob, annot=True, fmt='.3f')
plt.savefig('ConfMatrixProb.png')
