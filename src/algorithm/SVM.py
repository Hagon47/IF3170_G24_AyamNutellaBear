from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
        tolerance=1e-6,
        max_samples=None,
        random_state=42,
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.degree = degree
        self.coef0 = coef0
        self.multi_class = multi_class
        self.n_iters = n_iters
        self.tolerance = tolerance
        self.max_samples = max_samples
        self.random_state = random_state

        self.classes_ = None
        self.models_ = None   

        self.alpha = None
        self.b = None
        self.support_vectors = None
        self.w = None
        self.margin = None
        self.converged_ = False
        self.n_iterations_ = 0

    # Fit function to train the SVM model
    def fit(self, X, y, sample_weight=None):
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
        if self.max_samples is not None:
            if not isinstance(self.max_samples, (int, np.integer)) or self.max_samples < 2:
                raise ValueError("max_samples must be an integer >= 2")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("Invalid training data dimensions")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains invalid values")
        if y.shape[0] == 0:
            raise ValueError("Training data cannot be empty")
        if y.dtype.kind not in ('b', 'i', 'u', 'f', 'U', 'S', 'O'):
            raise ValueError("Invalid training labels")
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.ndim != 1 or sample_weight.shape[0] != X.shape[0]:
                raise ValueError("sample_weight must be a 1-D array matching X")
            if not np.all(np.isfinite(sample_weight)):
                raise ValueError("sample_weight contains invalid values")
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight cannot contain negative values")
            if not np.any(sample_weight > 0):
                raise ValueError("sample_weight must contain at least one positive value")
            sample_weight = sample_weight / np.mean(sample_weight)

        self.classes_ = np.unique(y)
        k = len(self.classes_)
        if k < 2:
            raise ValueError("SVM requires at least two classes")
        self.n_features_ = X.shape[1]

        if self.max_samples is not None and X.shape[0] > self.max_samples:
            rng = np.random.default_rng(self.random_state)
            selected = []
            proportions = [len(np.flatnonzero(y == cls)) / len(y) for cls in self.classes_]
            counts = [max(1, int(self.max_samples * proportion)) for proportion in proportions]
            counts = [
                min(count, len(np.flatnonzero(y == cls)))
                for cls, count in zip(self.classes_, counts)
            ]
            remainder = self.max_samples - sum(counts)
            fractional = [
                (self.max_samples * proportion) - count
                for proportion, count in zip(proportions, counts)
            ]
            for idx in np.argsort(fractional)[::-1][:max(0, remainder)]:
                counts[idx] += 1
            for cls, count in zip(self.classes_, counts):
                cls_indices = np.flatnonzero(y == cls)
                selected.extend(rng.choice(cls_indices, size=count, replace=False).tolist())
            selected = np.asarray(selected, dtype=int)
            rng.shuffle(selected)
            X = X[selected]
            y = y[selected]
            sample_weight = sample_weight[selected]
            self.classes_ = np.unique(y)
            k = len(self.classes_)
            if k < 2:
                raise ValueError("SVM requires at least two classes after sampling")

        if k == 2:
            y_bin = self._get_cls_map(y, self.classes_[1])
            self.models_ = [self._train_binary(X, y_bin, sample_weight)]

            m = self.models_[0]
            self.alpha = m['alpha']
            self.b = m['b']
            self.support_vectors = m['support_vectors']
            self.w = m['w']
            self.margin = m['margin']
            self.converged_ = bool(m['converged'])
            self.n_iterations_ = int(m['n_iterations'])

        elif self.multi_class == 'ovr':
            self.models_ = [
                self._train_binary(X, self._get_cls_map(y, c), sample_weight)
                for c in self.classes_
            ]

        else:
            self.models_ = {}
            for i in range(k):
                for j in range(i + 1, k):
                    mask = (y == self.classes_[i]) | (y == self.classes_[j])
                    y_bin = self._get_cls_map(y[mask], self.classes_[i])
                    self.models_[(i, j)] = self._train_binary(
                        X[mask], y_bin, sample_weight[mask]
                    )

        if k > 2:
            models = list(self.models_.values()) if isinstance(self.models_, dict) else self.models_
            self.converged_ = all(bool(model['converged']) for model in models)
            self.n_iterations_ = max(int(model['n_iterations']) for model in models)

        return self

    def _train_binary(self, X, y, sample_weight=None):
        n_samples = X.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape != (n_samples,):
                raise ValueError("sample_weight must match the number of training samples")
            if np.any(sample_weight < 0) or not np.all(np.isfinite(sample_weight)):
                raise ValueError("sample_weight must contain finite non-negative values")
            sample_weight = sample_weight / np.mean(sample_weight)
        K = self._compute_kernel(X, X)

        # Q_ij = y_i * y_j * K(x_i, x_j)
        Q = np.outer(y, y) * K
        Q = 0.5 * (Q + Q.T)  

        def objective(alpha):
            return 0.5 * alpha @ Q @ alpha - np.sum(alpha)

        def objective_jac(alpha):
            return Q @ alpha - np.ones_like(alpha)

        constraints = [{
            'type': 'eq',
            'fun': lambda alpha: np.dot(alpha, y),
            'jac': lambda alpha: y,
        }]

        upper_bound = None if np.isinf(self._C) else float(self._C)
        bounds = (
            [(0.0, upper_bound) for _ in range(n_samples)]
            if upper_bound is None
            else [(0.0, upper_bound * float(w)) for w in sample_weight]
        )
        initial_alpha = np.zeros(n_samples, dtype=float)

        result = minimize(
            fun=objective,
            x0=initial_alpha,
            method='SLSQP',
            jac=objective_jac,
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': int(self.n_iters),
                'ftol': max(float(self.tolerance), 1e-9),
                'disp': False,
            },
        )

        alpha = np.asarray(result.x, dtype=float)

        eps = max(1e-8, float(self.tolerance))
        alpha[np.abs(alpha) < eps] = 0.0
        if upper_bound is not None:
            upper_bounds = np.asarray([bound[1] for bound in bounds], dtype=float)
            alpha = np.clip(alpha, 0.0, upper_bounds)
            close_to_upper = np.abs(alpha - upper_bounds) < eps
            alpha[close_to_upper] = upper_bounds[close_to_upper]
        else:
            alpha = np.maximum(alpha, 0.0)

        support = alpha > eps
        upper_bounds = (
            np.asarray([bound[1] for bound in bounds], dtype=float)
            if upper_bound is not None
            else np.zeros(n_samples, dtype=float)
        )
        at_upper = (
            alpha >= upper_bounds - eps
            if upper_bound is not None
            else np.zeros(n_samples, dtype=bool)
        )

        decision_no_bias = K @ (alpha * y)
        free_sv = support & ~at_upper

        if np.any(free_sv):
            b = float(np.mean(y[free_sv] - decision_no_bias[free_sv]))
        else:
            t = y - decision_no_bias
            lower = ((y == 1) & ~support) | ((y == -1) & at_upper)
            upper = ((y == -1) & ~support) | ((y == 1) & at_upper)

            lo = float(np.max(t[lower])) if np.any(lower) else -np.inf
            hi = float(np.min(t[upper])) if np.any(upper) else np.inf

            if np.isfinite(lo) and np.isfinite(hi):
                b = (lo + hi) / 2.0
            elif np.isfinite(lo):
                b = lo
            elif np.isfinite(hi):
                b = hi
            else:
                b = 0.0

        model = {
            'alpha': alpha,
            'b': b,
            'support_vectors': X[support],
            'support_vector_alpha_y': alpha[support] * y[support],
            'w': None,
            'margin': None,
            'converged': bool(result.success),
            'n_iterations': int(result.nit),
            'optimizer_message': str(result.message),
            'dual_objective': float(-result.fun),
        }

        if self.kernel == 'linear':
            w = model['support_vector_alpha_y'] @ model['support_vectors']
            norm_w = np.linalg.norm(w)
            model['w'] = w
            model['margin'] = 2.0 / norm_w if norm_w > 0 else np.inf

        return model

    def _check_fitted(self, X):
        if self.models_ is None:
            raise ValueError("Model has not been trained")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError("Invalid feature dimensions")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains invalid values")
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

    @staticmethod
    def plot_decision_boundaries(
        X,
        y,
        save_dir,
        max_samples=700,
        seed=42,
        n_iters=250,
        tolerance=1e-5,
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("Invalid visualization data dimensions")

        rng = np.random.default_rng(seed)
        indices = []
        for label in np.unique(y):
            class_indices = np.flatnonzero(y == label)
            per_class = min(max_samples // len(np.unique(y)), len(class_indices))
            indices.extend(
                rng.choice(class_indices, size=per_class, replace=False).tolist()
            )
        indices = np.asarray(indices)
        X_vis = X[indices]
        y_vis = y[indices]

        if X_vis.shape[1] > 2:
            X_vis = PCA(n_components=2, random_state=seed).fit_transform(X_vis)
        elif X_vis.shape[1] < 2:
            raise ValueError("Decision-boundary visualization requires at least two features")

        plot_specs = [
            ('Linear', 'linear_decision_boundary.png', 'linear', {}),
            ('Polynomial', 'polynomial_decision_boundary.png', 'polynomial',
             {'degree': 2, 'coef0': 0.7}),
            ('Gaussian RBF', 'gaussian_rbf_decision_boundary.png', 'rbf',
             {'gamma': 0.05}),
        ]

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for title, filename, kernel, options in plot_specs:
            vis_model = Svm(
                kernel=kernel,
                C=1.0,
                n_iters=n_iters,
                tolerance=tolerance,
                **options,
            ).fit(X_vis, y_vis)

            x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
            y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 300),
                np.linspace(y_min, y_max, 300),
            )
            grid = np.column_stack([xx.ravel(), yy.ravel()])
            zz = vis_model.decision_function(grid).reshape(xx.shape)

            plt.figure(figsize=(8, 6))
            plt.contourf(
                xx, yy, zz,
                levels=[-np.inf, 0, np.inf],
                alpha=0.20,
                colors=['#8db4e2', '#f2b6b6'],
            )
            plt.contour(
                xx, yy, zz,
                levels=[-1, 0, 1],
                colors=['#1f4e79', 'black', '#9c0006'],
                linestyles=['--', '-', '--'],
                linewidths=[1.5, 2, 1.5],
            )
            plt.scatter(
                X_vis[:, 0], X_vis[:, 1],
                c=y_vis, cmap='coolwarm',
                edgecolors='k', s=25, alpha=0.75,
                label='Data latih',
            )
            plt.scatter(
                vis_model.support_vectors[:, 0],
                vis_model.support_vectors[:, 1],
                s=95, facecolors='none', edgecolors='gold',
                linewidths=1.5,
                label=f'Support vector ({len(vis_model.support_vectors)})',
            )
            plt.title(
                f'SVM {title}: bidang pemisah, margin, dan support vector'
            )
            plt.xlabel('PCA component 1')
            plt.ylabel('PCA component 2')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(save_dir / filename, dpi=150)
            plt.show()

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def dual_objective(self):
        if self.models_ is None:
            raise ValueError("Model has not been trained")

        values = []
        for model in self.models_ if isinstance(self.models_, list) else self.models_.values():
            alpha = model['alpha']
            sv_alpha_y = model['support_vector_alpha_y']
            support = model['support_vectors']
            if len(support) == 0:
                values.append(0.0)
                continue
            y_alpha = sv_alpha_y
            K = self._compute_kernel(support, support)
            values.append(float(np.sum(alpha) - 0.5 * np.sum(np.outer(y_alpha, y_alpha) * K)))
        return values[0] if len(values) == 1 else values

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
                'n_iterations': model.get('n_iterations', 0),
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
            tolerance=np.asarray(self.tolerance),
            max_samples=np.asarray(-1 if self.max_samples is None else self.max_samples),
            random_state=np.asarray(self.random_state),
            classes_=np.asarray(self.classes_),
            n_features_=np.asarray(self.n_features_),
            model_count=np.asarray(len(models)),
            models=np.asarray(models, dtype=object),
            model_keys=np.asarray(model_keys, dtype=object),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=True) as data:
            model = cls(
                kernel=str(data['kernel'].item()),
                gamma=float(data['gamma'].item()),
                C=None if not np.isfinite(float(data['C'].item())) else float(data['C'].item()),
                degree=int(data['degree'].item()),
                coef0=float(data['coef0'].item()),
                multi_class=str(data['multi_class'].item()),
                n_iters=int(data['n_iters'].item()),
                tolerance=float(data['tolerance'].item()),
                max_samples=None if int(data['max_samples'].item()) < 0 else int(data['max_samples'].item()),
                random_state=int(data['random_state'].item()),
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
                    'n_iterations': int(saved['n_iterations']) if 'n_iterations' in saved else 0,
                })

            if len(model.classes_) == 2:
                model.models_ = restored
                binary = restored[0]
                model.alpha = binary['alpha']
                model.b = binary['b']
                model.support_vectors = binary['support_vectors']
                model.w = binary['w']
                model.margin = binary['margin']
                model.converged_ = bool(binary['converged'])
                model.n_iterations_ = int(binary['n_iterations'])
            elif model.multi_class == 'ovr':
                model.models_ = restored
            else:
                model.models_ = {}
                for pair, saved in zip(model_keys, restored):
                    key = tuple(int(value) for value in pair)
                    model.models_[key] = saved
            if len(model.classes_) > 2:
                model.converged_ = all(bool(saved['converged']) for saved in restored)
                model.n_iterations_ = max(int(saved['n_iterations']) for saved in restored)

        return model