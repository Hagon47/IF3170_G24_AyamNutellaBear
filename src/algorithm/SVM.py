from abc import abstractmethod

import numpy as np

class Svm:
    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
    
    # Fit function to train the SVM model
    def fit(self, X, y):
        pass
    
    # Predict function to make predictions using the trained SVM model
    def predict(self, X):
        pass
    
    
        