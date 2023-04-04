import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from numpy import round

NUM_CLASSES = 10
NUM_FEATURES = 5
RANDOM_SEED = 42


X_blob, y_blob = make_blobs(n_samples=1000, n_features=NUM_FEATURES, centers=NUM_CLASSES, cluster_std=1.5,
                            random_state=RANDOM_SEED)

X_blob = torch.from_numpy(X_blob).type(torch.float32)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

X_train, X_test, y_train, y_test = train_test_split(X_blob, y_blob, test_size=0.2, random_state=RANDOM_SEED)

# plt.figure(figsize=(10, 7))
# ax = plt.axes(projection="3d")
# ax.scatter3D(X_blob[:, 0], X_blob[:, 1], X_blob[:, 2], c=y_blob, cmap=plt.cm.RdYlBu)
# plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"


class BlobModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_units=8):
        """ Initializes multi-class classification model
        Args:
            in_features (int): Number of input features to the model
            out_features (int): Number of outputs features (classes)
            hidden_units (int): Number of hidden units between layers, default = 8
        """

        super().__init__()
        self.lin_layers_stack = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=out_features)
        )

    def forward(self, x):
        return self.lin_layers_stack(x)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # for pytorch it works
    acc = (correct / len(y_pred)) * 100
    return acc

model_0 = BlobModel(in_features=NUM_FEATURES, out_features=NUM_CLASSES).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)


epochs = 300

X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)

for epoch in range(1, epochs + 1):
    model_0.train()

    y_logits = model_0(X_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train,y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test)
        test_pred = torch.softmax(test_logits,dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)

    if epoch % 100 == 0:
        print(f'Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}%, | Test loss: {test_loss:.5f}'
              f', Test acc: {test_acc:.2f}%')

# Make predictions
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test)

y_pred_probs = torch.softmax(y_logits, dim=1)
y_preds = torch.argmax(y_pred_probs, dim=1)

# plt.figure(figsize=(10, 7))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_0, X_train, y_train)
# plt.subplot(1, 2, 2)
# plt.title('Test')
# plot_decision_boundary(model_0, X_test,y_test)
# plt.show()


# METRICS

torch_accuracy = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(device)
torch_precision = Precision(task="multiclass", average='macro', num_classes=NUM_CLASSES).to(device)
torch_recall = Recall(task="multiclass", average='macro', num_classes=NUM_CLASSES).to(device)
torch_f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES).to(device)
torch_confmat = ConfusionMatrix(task="multiclass", num_classes=NUM_CLASSES).to(device)


accuracy = torch_accuracy(y_preds, y_test)
precision = torch_precision(y_preds, y_test)
recall = torch_recall(y_preds, y_test)
f1 = torch_f1(y_preds, y_test)
conf_matrix = torch_confmat(y_preds, y_test)

print(f'Точность (acc) составляет: {accuracy} | Точность составляет: '
      f'{precision} | Полнота составляет: {recall} ')
print(f'F-мера составляет: {f1} | Матрица ошибок имеет следующий вид:')
print(conf_matrix)