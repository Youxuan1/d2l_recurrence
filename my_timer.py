"""Timer utility class for measuring runtime."""

import time

import numpy as np


class Timer:
    """Record multiple runtime"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self) -> None:
        """Init timer"""
        self.tik = time.time()

    def stop(self) -> float:
        """Stop timer and record it"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self) -> float:
        """Return average runtime"""
        return sum(self.times) / len(self.times)

    def sum(self) -> float:
        """Return total runtime"""
        return sum(self.times)

    def cumsum(self) -> float:
        """Return accumulative runtime"""
        return np.array(self.times).cumsum().tolist()


if __name__ == "__main__":
    import torch

    N = int(1e4)
    a = torch.ones([N])
    b = torch.ones([N])
    c = torch.zeros(N)
    timer = Timer()
    for i in range(N):
        c[i] = a[i] + b[i]
    print(f"{timer.stop():.5f} sec")
