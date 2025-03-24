"""Define a class to generate data for linear regression."""

import numpy as np
import torch
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D


class LinearRegressionDataGenerator:
    """Class to generate data for linear regression."""

    def __init__(
        self, w: np.ndarray[float], b: np.ndarray[float], noise_std: float = 0.01
    ):
        """
        Initialize generator, w.r.t. weight, bias and noise std.
        """
        self.w = w
        self.b = b
        self.noise_std = noise_std
        self.x = None
        self.y = None

    def synthetic_data(
        self, num_examples: float
    ) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Generate data: y = Xw + b + noise
        """
        self.x = torch.normal(0, 1, (num_examples, len(self.w)))
        self.y = torch.mm(self.x, self.w) + self.b
        self.y += torch.normal(0, self.noise_std, self.y.shape)

        return self.x, self.y.reshape((-1, 1))

    def visualize(self) -> None:
        """
        Visualize generated data. Works when feature dimension is 1 or 2.
        """
        if self.x is None or self.y is None:
            raise ValueError("Please generate data first using synthetic_data().")

        if self.x.shape[1] == 1:
            # 一维特征，画二维图
            plt.figure(figsize=(8, 5))
            plt.scatter(self.x.numpy(), self.y.numpy(), alpha=0.6)
            plt.title("Linear Regression Data (1D)")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.show()
        elif self.x.shape[1] == 2:
            # 二维特征，画三维图

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                self.x[:, 0].numpy(), self.x[:, 1].numpy(), self.y.numpy(), alpha=0.6
            )
            ax.set_title("Linear Regression Data (2D)")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("y")
            plt.show()
        else:
            print(
                f"⚠️ Cannot visualize data with {self.x.shape[1]}-dimensional features."
            )
