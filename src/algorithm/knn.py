import numpy as np


class KNN:
    def __init__(self, k=3, p=2):
        if type(k) is not int or k < 1:
            raise ValueError("k harus bilangan bulat positif.")
        self.k = k
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.X_train = X
        self.y_train = y
        return self

    def _euclidean_distances(self, x):
        differences = self.X_train - x
        return np.sqrt(np.sum(differences ** 2, axis=1))

    def _minkowski_distances(self, x):
        differences = np.abs(self.X_train - x)
        return np.sum(differences ** self.p, axis=1) ** (1 / self.p)

    def predict(self, X):
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model belum di-fit.")

        predict = []

        for x in X:
            distance = 0
            if self.p == 2 :
                distance = self._euclidean_distances(x) 
            else :
                distance = self._minkowski_distances(x)

            top_neighbor = np.argsort(distance, kind="stable")[:self.k]
            label = self.y_train[top_neighbor]
            labels, count = np.unique(label, return_counts=True)
            predict.append(labels[np.argmax(count)])
        return np.array(predict)

