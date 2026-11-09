from abc import abstractmethod

import numpy as np

class Svm:
    def __init__(self, kernel='linear', learning_rate=0.01, n_iters=1000):
        self.kernel = kernel
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    # Fit function to train the SVM model
    def fit(self, X, y):
        pass
    
    # Predict function to make predictions using the trained SVM model
    def predict(self, X):
        pass
    
    
        