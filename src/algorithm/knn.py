import numpy as np


class KNN:
    def __init__(self, k=3):
        if type(k) is not int or k < 1:
            raise ValueError("k harus bilangan bulat positif.")
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)

        if X.ndim != 2 or X.size == 0:
            raise ValueError("X harus matriks 2D yang tidak kosong.")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y harus vektor 1D dengan panjang yang sama dengan jumlah baris X.")
        if self.k > len(X):
            raise ValueError("k tidak boleh lebih besar dari jumlah baris X.")

        self.X_train = X
        self.y_train = y
        return self