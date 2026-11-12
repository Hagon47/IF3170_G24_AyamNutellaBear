import numpy as np
from scipy.optimize import minimize

class Svm:
    def __init__(self, kernel='linear', gamma=0.5, C=1.0, n_iters=1000, learning_rate=1e-3, tolerance=1e-6):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.n_iters = n_iters

        self.alpha = None
        self.b = None
        self.X_train = None
        self.y_train = None
    
    # Fit function to train the SVM model
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = self._get_cls_map(y)
        
        self.X_train = X
        self.y_train = y
        
        n_samples= X.shape[0]

        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        
        K = self._compute_kernel(X, X) # K_ij = K(X_i, X_j)

        Q = np.outer(y, y) * K # Qij = y_i * y_j * K(X_i, X_j)
        
        # Gradient Descent
        for _ in range(self.n_iters):
            gradient = self._gradient(Q, self.alpha)
            
            alpha_new = (
                self.alpha - self.learning_rate * gradient
            )

            alpha_new = self._project_alpha(alpha_new, X)
            
            difference = np.linalg.norm(alpha_new - self.alpha)
            
            self.alpha = alpha_new
            
            if difference < self.tolerance:
                break

        
        # bias
        decision = K @ (self.alpha * y) # decision = sum(alpha_i * y_i * K(X_i, X)) for all i
        
        free_sv = (self.alpha > 1e-5) & (self.alpha < self.C - 1e-5)

        if np.any(free_sv):
            self.b = np.mean(y[free_sv] - decision[free_sv])
        else: 
            support_vectors = self.alpha > 1e-5
            
            if np.any(support_vectors):
                self.b = np.mean(y[support_vectors] - decision[support_vectors])
                
        return self
            
    def _project_alpha(self, alpha, X):
        low = -(np.max(np.abs(alpha)) + self.C + 1)
        high = np.max(np.abs(alpha)) + self.C + 1

        y = self.y_train
        
        for _ in range(100):
            mid = (low + high) / 2

            projected = np.clip(
                alpha - mid * y,
                0,
                self.C
            )

            constraint = np.dot(projected, y)

            if constraint > 0:
                low = mid
            else:
                high = mid

        return projected
            
            
    # Inference function to predict new data points based on the trained SVM model
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        
        K = self._compute_kernel(X, self.X_train) # K_ij = K(X_i, X_j)
        
        decision = K @ (self.alpha * self.y_train) + self.b 
        
        return decision
        
    # Function to compute based on the selected kernel type
    def _compute_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'polynomial':
            return (np.dot(X1, X2.T) + 1) ** 2
        elif self.kernel == 'rbf':
            distance =  (
                np.sum(X1 ** 2, axis=1)[:, None]
                + np.sum(X2 ** 2, axis=1)[None, :]
                - 2 * np.dot(X1, X2.T)
            )
            
            return np.exp(-self.gamma * np.maximum(distance, 0))
        else:
            raise ValueError("Unsupported kernel type")
        
    def _get_cls_map(self, y):
        return np.where(np.asarray(y) <= 0, -1, 1)
    
    @staticmethod
    # Objective function for the dual problem of SVM
    def _objective( alpha, Q):
        return (
            0.5 * alpha @ Q @ alpha
            - np.sum(alpha)
        )
        
    @staticmethod
    # Gradient of the objective function for the dual problem of SVM
    def _gradient(Q, alpha):
        return Q @ alpha - np.ones_like(alpha)