from pathlib import Path
import numpy as np

class LogisticRegression:
    # Rumus penalti L1, L2, dan Elastic Net dari 
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    
    def __init__(self, learning_rate=0.001, max_iteration = 100, penalty = None, lmda = 0.01, l1_ratio = 0.5, random = 7,shuffle = True):
        if learning_rate <=0 or np.isfinite(learning_rate) == False:
            raise ValueError("Learning rate harus positif atau tidak tak hingga.")
        if not isinstance(max_iteration, (int, np.integer)) or max_iteration < 1:
            raise ValueError("Iterasi maksimum harus bilangan bulat positif.")
        if penalty not in [None, 'l1', 'l2', 'elasticnet', 'none', 'L1', 'L2', 'ELASTICNET', 'NONE']:
            raise ValueError("Penalty harus None, 'l1', 'l2', atau 'elasticnet'.")
        if lmda < 0 or np.isfinite(lmda) == False:
            raise ValueError("Lambda harus bilangan positif atau nol.")
        if not (0 <= l1_ratio <= 1):
            raise ValueError("Ratio l1 harus antara 0 dan 1.")
        
        self.learning_rate = float(learning_rate)
        self.max_iteration = int(max_iteration)
        self.penalty = penalty
        self.lmda = float(lmda)
        self.l1_ratio = float(l1_ratio)
        self.random = int(random)
        self.shuffle = bool(shuffle)
        
        self.b = None
        self.loss_history=[]
        self.n_iteration = 0
        
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def is_fitted(self):
        if self.b is None:
            raise ValueError("Panggil metode fit() terlebih dahulu.")
        
    def regularization_loss(self):
        w = self.b[1:] 
        
        if self.penalty is None or self.penalty.lower() == 'none':
            return 0
        elif self.penalty.lower() == 'l1':
            return self.lmda * np.sum(np.abs(w))
        elif self.penalty.lower() == 'l2':
            return self.lmda/2 * np.sum(w ** 2)
        
        l1 = np.sum(np.abs(w))
        l2 = np.sum(w**2) / 2
        return self.lambda_ * (self.l1_ratio * l1 + (1 - self.l1_ratio) * l2)
    
    
    
    
        
        
        
        