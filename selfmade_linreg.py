"""From zero to implement Linear Regression"""

import random
import torch
from d2l import torch as d2l


def synthetic_data(w, b, num_examples):
    """Generate y=Xw+b+noise"""
    num_traits = len(w)
    X = torch.normal(0, 1, (num_examples, num_traits))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 1, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    """Shuffle samples in dataset, then offer a small batch of data."""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """Return a linear regression model"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """Return MSE"""
    return 0.5 * (y.reshape(y_hat.shape) - y_hat) ** 2


def sgd(params, lr, batch_size):
    """Small batch stochastic gradient descending"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # for i in range(len(features)):
    #     print(
    #         "sample ",
    #         i,
    #         ", feature: ",
    #         features[i].detach().tolist(),
    #         ", label: ",
    #         labels[i].detach().tolist(),
    #     )

    # d2l.set_figsize()
    # d2l.plt.scatter(
    #     features[:, 1].detach().numpy(), labels.detach().numpy(), 1, c="tab:red"
    # )
    # d2l.plt.show()

    batch_size = 10

    # for X, y in data_iter(batch_size, features, labels):
    #     print(
    #         "X.shape = ",
    #         X.size(),  # Equals to X.shape
    #         "\n",
    #         X.detach().numpy(),
    #         "\n",
    #         "y.shape = ",
    #         y.shape,
    #         "\n",
    #         y.detach().numpy(),
    #     )
    #     break

    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    lr = 0.000003
    num_epochs = 30000
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)

        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")
