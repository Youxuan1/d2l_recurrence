"""Define a class to generate data for linear regression."""

import numpy as np
from numpy.typing import NDArray
import torch
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D


class LinearRegressionDataGenerator:
    """Class to generate data for linear regression."""

    def __init__(
        self,
        weight: NDArray[np.float32],
        bias: NDArray[np.float32],
        noise_std: float = 0.01,
    ):
        """
        Initialize generator, w.r.t. weight, bias and noise std.
        """
        self.weight = torch.tensor(weight, dtype=torch.float32)
        self.bias = torch.tensor(bias, dtype=torch.float32)
        self.noise_std = noise_std
        self.x = None
        self.y = None

    def synthetic_data(
        self, num_examples: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Generate data: y = Xw + b + noise
        """
        self.x = torch.normal(0, 1, (num_examples, len(self.weight)))
        self.y = torch.matmul(self.x, self.weight) + self.bias
        self.y += torch.normal(0, self.noise_std, self.y.shape)

        return (
            self.x.cpu().detach().numpy(),
            self.y.reshape((-1, 1)).cpu().detach().numpy(),
        )

    def visualize(self) -> None:
        """
        Visualize generated data. Works when feature dimension is 1 or 2.
        """
        if self.x is None or self.y is None:
            raise ValueError("Please generate data first using synthetic_data().")

        x_np = self.x.cpu().detach().numpy()
        y_np = self.y.cpu().detach().numpy()

        if x_np.shape[1] == 1:
            # 一维特征，画二维图
            plt.figure(figsize=(8, 5))
            plt.scatter(x_np, y_np, alpha=0.6)
            plt.title("Linear Regression Data (1D)")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.show()
        elif x_np.shape[1] == 2:
            # 二维特征，画三维图

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(x_np[:, 0], x_np[:, 1], y_np, alpha=0.6)
            ax.set_title("Linear Regression Data (2D)")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("y")
            plt.show()
        else:
            print(f"⚠️ Cannot visualize data with {x_np.shape[1]}-dimensional features.")


if __name__ == "__main__":
    # 测试 1D
    w = np.array([2.0])  # 改成 [2.0, -3.4] 测试 2D
    B = 4.2

    generator = LinearRegressionDataGenerator(w, B, noise_std=0.01)
    X, y = generator.synthetic_data(1000)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    generator.visualize()

    # 测试 2D 特征数据
    w = np.array([2.0, -3.4])  # 2个特征
    B = 4.2

    generator = LinearRegressionDataGenerator(w, B, noise_std=0.01)
    X, y = generator.synthetic_data(1000)

    print("✅ 2D 测试")
    print("X shape:", X.shape)  # (1000, 2)
    print("y shape:", y.shape)  # (1000, 1)

    generator.visualize()
