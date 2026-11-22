import warnings
from pathlib import Path

import numpy as np


class Svm:
    KERNELS = ('linear', 'polynomial', 'rbf', 'sigmoid')
    MULTI_CLASS = ('ovr', 'ovo', 'dag')

    def __init__(
        self,
        kernel='linear',
        gamma=0.5,
        C=1.0,
        degree=3,
        coef0=1.0,
        multi_class='ovo',
        n_iters=1000,
        learning_rate=1e-3,
        tolerance=1e-6,
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.degree = degree
        self.coef0 = coef0
        self.multi_class = multi_class
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.tolerance = tolerance

        self.classes_ = None
        self.models_ = None   

        self.alpha = None
        self.b = None
        self.support_vectors = None
        self.w = None
        self.margin = None

    # Fit function to train the SVM model
    def fit(self, X, y):
        if self.kernel not in self.KERNELS:
            raise ValueError("Unsupported kernel type")
        if self.multi_class not in self.MULTI_CLASS:
            raise ValueError(f"multi_class harus salah satu dari {self.MULTI_CLASS}")

        self._C = np.inf if self.C is None else float(self.C)
        if self._C <= 0:
            raise ValueError("C must be positive")
        if self.kernel != 'linear' and self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.kernel == 'polynomial' and (
            int(self.degree) != self.degree or self.degree < 1
        ):
            raise ValueError("degree must be an integer >= 1")
        if not isinstance(self.n_iters, (int, np.integer)) or self.n_iters < 1:
            raise ValueError("n_iters must be a positive integer")
        if self.learning_rate is not None and (
            not np.isfinite(self.learning_rate) or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive or None")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("Invalid training data dimensions")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains invalid values")
        if y.shape[0] == 0:
            raise ValueError("Training data cannot be empty")

        self.classes_ = np.unique(y)
        k = len(self.classes_)
        if k < 2:
            raise ValueError("SVM requires at least two classes")
        self.n_features_ = X.shape[1]

        if k == 2:
            y_bin = self._get_cls_map(y, self.classes_[1])
            self.models_ = [self._train_binary(X, y_bin)]

            m = self.models_[0]
            self.alpha = m['alpha']
            self.b = m['b']
            self.support_vectors = m['support_vectors']
            self.w = m['w']
            self.margin = m['margin']

        elif self.multi_class == 'ovr':
            self.models_ = [
                self._train_binary(X, self._get_cls_map(y, c))
                for c in self.classes_
            ]

        else:
            self.models_ = {}
            for i in range(k):
                for j in range(i + 1, k):
                    mask = (y == self.classes_[i]) | (y == self.classes_[j])
                    y_bin = self._get_cls_map(y[mask], self.classes_[i])
                    self.models_[(i, j)] = self._train_binary(X[mask], y_bin)

        return self

    def _train_binary(self, X, y):
        n_samples = X.shape[0]

        alpha = np.zeros(n_samples)

        K = self._compute_kernel(X, X)          # K_ij = K(X_i, X_j)
        Q = np.outer(y, y) * K                  # Qij = y_i * y_j * K(X_i, X_j)
        Q = (Q + Q.T) / 2

        if self.learning_rate is None:
            lr = 1.0 / max(np.linalg.eigvalsh(Q)[-1], 1e-12)
        else:
            lipschitz_limit = 1.0 / max(np.linalg.eigvalsh(Q)[-1], 1e-12)
            lr = min(float(self.learning_rate), lipschitz_limit)

        converged = False
        for _ in range(self.n_iters):
            gradient = self._gradient(Q, alpha)

            alpha_new = alpha - lr * gradient
            alpha_new = self._project_alpha(alpha_new, y)

            difference = np.linalg.norm(alpha_new - alpha)
            alpha = alpha_new

            if difference < self.tolerance:
                converged = True
                break

        if not converged:
            warnings.warn(
                "SVM belum konvergen; naikkan n_iters / learning_rate "
                "atau gunakan learning_rate=None.",
                RuntimeWarning,
            )

        eps = 1e-5 * (self._C if np.isfinite(self._C) else 1.0)
        support = alpha > eps
        at_upper = (alpha >= self._C - eps) if np.isfinite(self._C) \
            else np.zeros(n_samples, dtype=bool)
        free_sv = support & ~at_upper           # 0 < alpha < C

        decision = K @ (alpha * y)

        if np.any(free_sv):
            b = float(np.mean(y[free_sv] - decision[free_sv]))
        else:
            t = y - decision
            lower = ((y == 1) & ~support) | ((y == -1) & at_upper)
            upper = ((y == -1) & ~support) | ((y == 1) & at_upper)
            lo = t[lower].max() if np.any(lower) else -np.inf
            hi = t[upper].min() if np.any(upper) else np.inf

            if np.isfinite(lo) and np.isfinite(hi):
                b = float((lo + hi) / 2)
            elif np.isfinite(lo):
                b = float(lo)
            elif np.isfinite(hi):
                b = float(hi)
            else:
                b = 0.0

        model = {
            'alpha': alpha,
            'b': b,
            'support_vectors': X[support],
            'support_vector_alpha_y': alpha[support] * y[support],
            'w': None,
            'margin': None,
            'converged': converged,
        }

        if self.kernel == 'linear':
            w = model['support_vector_alpha_y'] @ model['support_vectors']
            norm_w = np.linalg.norm(w)
            model['w'] = w
            model['margin'] = 2.0 / norm_w if norm_w > 0 else np.inf

        return model

    def _project_alpha(self, alpha, y):
        def get_projected(lam):
            return np.clip(alpha - lam * y, 0, self._C)

        def constraint(lam):                   
            return np.dot(get_projected(lam), y)

        low = -(np.max(np.abs(alpha)) + 1.0)
        high = np.max(np.abs(alpha)) + 1.0

        for _ in range(100):
            if constraint(low) >= 0:
                break
            low *= 2
        for _ in range(100):
            if constraint(high) <= 0:
                break
            high *= 2

        for _ in range(100):
            mid = (low + high) / 2

            if constraint(mid) > 0:
                low = mid
            else:
                high = mid

            if high - low < 1e-12:
                break

        return get_projected((low + high) / 2)

    def _check_fitted(self, X):
        if self.models_ is None:
            raise ValueError("Model has not been trained")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError("Invalid feature dimensions")
        return X

    def _binary_decision(self, model, X):
        K = self._compute_kernel(X, model['support_vectors'])
        return K @ model['support_vector_alpha_y'] + model['b']

    def decision_function(self, X):
        X = self._check_fitted(X)
        if len(self.classes_) == 2:
            return self._binary_decision(self.models_[0], X)
        if self.multi_class == 'ovr':
            return np.column_stack(
                [self._binary_decision(m, X) for m in self.models_]
            )
        return {p: self._binary_decision(m, X) for p, m in self.models_.items()}

    def predict(self, X):
        X = self._check_fitted(X)
        k = len(self.classes_)
        scores = self.decision_function(X)

        if k == 2:
            return np.where(scores >= 0, self.classes_[1], self.classes_[0])

        if self.multi_class == 'ovr':       
            return self.classes_[np.argmax(scores, axis=1)]

        if self.multi_class == 'ovo':       
            votes = np.zeros((X.shape[0], k))
            for (i, j), f in scores.items():
                win_i = f >= 0
                votes[win_i, i] += 1
                votes[~win_i, j] += 1
            return self.classes_[np.argmax(votes, axis=1)]

        pred = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for n in range(X.shape[0]):
            cand = list(range(k))
            while len(cand) > 1:
                i, j = cand[0], cand[-1]
                if scores[(i, j)][n] >= 0:
                    cand.pop()              
                else:
                    cand.pop(0)            
            pred[n] = self.classes_[cand[0]]
        return pred

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    @property
    def n_models_(self):
        if self.models_ is None:
            raise ValueError("Model has not been trained")
        return len(self.models_)

    def _compute_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)

        elif self.kernel == 'polynomial':
            return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree

        elif self.kernel == 'rbf':
            distance = (
                np.sum(X1 ** 2, axis=1)[:, None]
                + np.sum(X2 ** 2, axis=1)[None, :]
                - 2 * np.dot(X1, X2.T)
            )
            return np.exp(-self.gamma * np.maximum(distance, 0))

        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(X1, X2.T) + self.coef0)

        else:
            raise ValueError("Unsupported kernel type")

    @staticmethod
    def _get_cls_map(y, positive_class):
        return np.where(np.asarray(y) == positive_class, 1.0, -1.0)

    @staticmethod
    def _objective(alpha, Q):
        return 0.5 * alpha @ Q @ alpha - np.sum(alpha)

    @staticmethod
    def _gradient(Q, alpha):
        return Q @ alpha - np.ones_like(alpha)

    def save(self, path):
        if self.models_ is None:
            raise ValueError("Model has not been trained")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        models = []
        model_keys = []
        for model in self.models_ if isinstance(self.models_, list) else self.models_.values():
            models.append({
                'alpha': model['alpha'],
                'b': model['b'],
                'support_vectors': model['support_vectors'],
                'support_vector_alpha_y': model['support_vector_alpha_y'],
                'w': model['w'] if model['w'] is not None else np.array([], dtype=float),
                'margin': model['margin'] if model['margin'] is not None else np.nan,
                'converged': model['converged'],
            })
        if isinstance(self.models_, dict):
            model_keys = [np.asarray(pair, dtype=np.int64) for pair in self.models_.keys()]

        np.savez_compressed(
            path,
            kernel=np.asarray(self.kernel),
            gamma=np.asarray(self.gamma),
            C=np.asarray(np.inf if self.C is None else self.C),
            degree=np.asarray(self.degree),
            coef0=np.asarray(self.coef0),
            multi_class=np.asarray(self.multi_class),
            n_iters=np.asarray(self.n_iters),
            learning_rate=np.asarray(self.learning_rate),
            tolerance=np.asarray(self.tolerance),
            classes_=np.asarray(self.classes_),
            n_features_=np.asarray(self.n_features_),
            model_count=np.asarray(len(models)),
            models=np.asarray(models, dtype=object),
            model_keys=np.asarray(model_keys, dtype=object),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=True) as data:
            learning_rate_value = data['learning_rate'].item()
            if learning_rate_value is None:
                learning_rate_value = None
            else:
                learning_rate_value = float(learning_rate_value)
            model = cls(
                kernel=str(data['kernel'].item()),
                gamma=float(data['gamma'].item()),
                C=None if not np.isfinite(float(data['C'].item())) else float(data['C'].item()),
                degree=int(data['degree'].item()),
                coef0=float(data['coef0'].item()),
                multi_class=str(data['multi_class'].item()),
                n_iters=int(data['n_iters'].item()),
                learning_rate=learning_rate_value,
                tolerance=float(data['tolerance'].item()),
            )
            model.classes_ = data['classes_']
            model.n_features_ = int(data['n_features_'].item())
            models = data['models'].tolist()
            model._C = np.inf if model.C is None else float(model.C)
            model_keys = data['model_keys'].tolist()

            restored = []
            for saved in models:
                margin = saved['margin'].item() if hasattr(saved['margin'], 'item') else saved['margin']
                restored.append({
                    'alpha': saved['alpha'],
                    'b': saved['b'],
                    'support_vectors': saved['support_vectors'],
                    'support_vector_alpha_y': saved['support_vector_alpha_y'],
                    'w': saved['w'] if saved['w'].size else None,
                    'margin': None if np.isnan(margin) else margin,
                    'converged': saved['converged'],
                })

            if len(model.classes_) == 2:
                model.models_ = restored
                binary = restored[0]
                model.alpha = binary['alpha']
                model.b = binary['b']
                model.support_vectors = binary['support_vectors']
                model.w = binary['w']
                model.margin = binary['margin']
            elif model.multi_class == 'ovr':
                model.models_ = restored
            else:
                model.models_ = {}
                for pair, saved in zip(model_keys, restored):
                    key = tuple(int(value) for value in pair)
                    model.models_[key] = saved

        return model