import torch
from torch import nn  # nn - blocks for nn
import matplotlib.pyplot as plt
import numpy as np


# FINAL WORKFLOW

# setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

weight = 0.2
bias = - 0.3

start = 0
end = 10
step = 0.1

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

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
epochs = 400
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    model_1.train()

    # Forward pass
    y_pred = model_1(X_train)

    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()

    # Loss backward
    loss.backward()

    # Step the optimizer
    optimizer.step()


    model_1.eval()

    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred, y_test)

    if epoch % 50 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")


print("The model learned the following values for weights and bias:")
print(list(model_1.parameters()))
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")


model_1.eval()
with torch.inference_mode():
    y_preds = model_1(X_test)

# plot_pred(predictions=y_preds.cpu())

# Saving

torch.save(model_1.state_dict(),'final_workflow.pth')

# Loading
loaded_model_1 = LinRegModelV2()
loaded_model_1.load_state_dict(torch.load('final_workflow.pth'))

loaded_model_1.to(device)

loaded_model_1.eval()
with torch.inference_mode():
    loaded_preds = loaded_model_1(X_test)
print(y_preds == loaded_preds)
