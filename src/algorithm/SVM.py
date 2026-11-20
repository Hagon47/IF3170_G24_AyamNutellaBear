
import numpy as np


class Svm:
    def __init__(
        self,
        kernel='linear',
        gamma=0.5,
        C=1.0,
        n_iters=1000,
        learning_rate=1e-3,
        tolerance=1e-6
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.tolerance = tolerance

        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None

        self.support_vectors = None
        self.support_vector_labels = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = self._get_cls_map(y)

        if X.ndim != 2 or X.shape[0] != len(y):
            raise ValueError("Invalid training data dimensions")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains invalid values")

        if self.C <= 0:
            raise ValueError("C must be positive")

        if len(np.unique(y)) != 2:
            raise ValueError("SVM requires two classes")

        self.X_train = X
        self.y_train = y

        n_samples = X.shape[0]

        # Compute kernel matrix
        K = self._compute_kernel(X, X)

        # Qij = yi * yj * K(Xi, Xj)
        Q = np.outer(y, y) * K

        # Ensure numerical symmetry
        Q = (Q + Q.T) / 2

        # Lipschitz constant of the gradient
        eigenvalues = np.linalg.eigvalsh(Q)
        L = max(eigenvalues[-1], 1e-12)

        # Initialize alpha
        alpha = np.zeros(n_samples)

        # FISTA initialization
        z = alpha.copy()
        t = 1.0

        # Accelerated Projected Gradient Descent
        for _ in range(self.n_iters):

            # Gradient at the accelerated point
            gradient = self._gradient(Q, z)

            # Gradient descent update
            alpha_new = z - gradient / L

            # Project alpha onto the feasible region
            alpha_new = self._project_alpha(alpha_new)

            # Check convergence
            difference = np.linalg.norm(alpha_new - alpha)

            if difference < self.tolerance:
                alpha = alpha_new
                break

            # FISTA acceleration
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2

            z = alpha_new + (
                (t - 1) / t_new
            ) * (alpha_new - alpha)

            alpha = alpha_new
            t = t_new

        self.alpha = alpha

        sv_mask = self.alpha > 1e-5

        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]

        free_sv = (
            (self.alpha > 1e-5) &
            (self.alpha < self.C - 1e-5)
        )

        decision = K @ (self.alpha * y)

        if np.any(free_sv):
            self.b = np.mean(
                y[free_sv] - decision[free_sv]
            )

        elif np.any(sv_mask):
            self.b = np.mean(
                y[sv_mask] - decision[sv_mask]
            )

        else:
            self.b = 0.0

        return self

    # Projection onto the feasible region
    def _project_alpha(self, alpha):
        y = self.y_train

        def get_projected(lam):
            return np.clip(
                alpha - lam * y,
                0,
                self.C
            )

        def constraint(lam):
            return np.dot(get_projected(lam), y)

        low = -1.0
        high = 1.0

        while constraint(low) < 0:
            low *= 2

        while constraint(high) > 0:
            high *= 2

        for _ in range(100):
            mid = (low + high) / 2

            if constraint(mid) > 0:
                low = mid
            else:
                high = mid

        return get_projected((low + high) / 2)

    def predict(self, X):
        if self.alpha is None:
            raise ValueError("Model has not been trained")

        X = np.asarray(X, dtype=float)

        K = self._compute_kernel(X, self.X_train)

        decision = (
            K @ (self.alpha * self.y_train) + self.b
        )

        return np.where(decision >= 0, 1, -1)

    # Compute kernel matrix
    def _compute_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)

        elif self.kernel == 'polynomial':
            return (np.dot(X1, X2.T) + 1) ** 2

        elif self.kernel == 'rbf':
            distance = (
                np.sum(X1 ** 2, axis=1)[:, None]
                + np.sum(X2 ** 2, axis=1)[None, :]
                - 2 * np.dot(X1, X2.T)
            )

            return np.exp(
                -self.gamma * np.maximum(distance, 0)
            )

        else:
            raise ValueError("Unsupported kernel type")

    def _get_cls_map(self, y):
        return np.where(np.asarray(y) <= 0, -1, 1)

    @staticmethod
    def _objective(alpha, Q):
        return (
            0.5 * alpha @ Q @ alpha
            - np.sum(alpha)
        )

    @staticmethod
    def _gradient(Q, alpha):
        return Q @ alpha - np.ones_like(alpha)
