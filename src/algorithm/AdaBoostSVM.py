import numpy as np

from .SVM import Svm


class AdaBoostSvm:
    def __init__(
        self,
        n_estimators=50,
        learning_rate=1.0,
        kernel='rbf',
        C=1.0,
        gamma='scale',
        degree=3,
        coef0=1.0,
        n_iters=1000,
        tolerance=1e-6,
        multi_class='ovo',
        max_samples=None,
        random_state=42,
    ):
        if not isinstance(n_estimators, (int, np.integer)) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_iters = n_iters
        self.tolerance = tolerance
        self.multi_class = multi_class
        self.max_samples = max_samples
        self.random_state = random_state

        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.sample_weight_history_ = []
        self.sample_weights_ = None
        self.classes_ = None
        self.n_features_ = None
        self.converged_ = False
        self.stopped_early_ = False
        self.stop_reason_ = None
        self.n_estimators_ = 0

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("Invalid training data dimensions")
        if X.shape[0] == 0:
            raise ValueError("Training data cannot be empty")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains invalid values")

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("AdaBoostSvm requires at least two classes")
        self.n_features_ = X.shape[1]

        n_samples = X.shape[0]
        if sample_weight is None:
            sample_weights = np.full(n_samples, 1.0 / n_samples, dtype=float)
        else:
            sample_weights = np.asarray(sample_weight, dtype=float)
            if sample_weights.ndim != 1 or sample_weights.shape[0] != n_samples:
                raise ValueError("sample_weight has invalid dimensions")
            if not np.all(np.isfinite(sample_weights)) or np.any(sample_weights < 0):
                raise ValueError("sample_weight must contain finite non-negative values")
            if not np.any(sample_weights > 0):
                raise ValueError("sample_weight must contain at least one positive value")
            sample_weights = sample_weights / np.sum(sample_weights)

        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.sample_weight_history_ = []
        self.converged_ = False
        self.stopped_early_ = False
        self.stop_reason_ = None

        threshold = 0.5 if n_classes == 2 else 1.0 - 1.0 / n_classes
        gamma = self.gamma
        if gamma == 'scale':
            feature_variance = float(np.var(X))
            gamma = 1.0 / (X.shape[1] * feature_variance) if feature_variance > 0 else 1.0

        for iteration in range(self.n_estimators):
            estimator_random_state = (
                None if self.random_state is None
                else int(self.random_state) + iteration
            )
            estimator = Svm(
                kernel=self.kernel,
                C=self.C,
                gamma=gamma,
                degree=self.degree,
                coef0=self.coef0,
                n_iters=self.n_iters,
                tolerance=self.tolerance,
                multi_class=self.multi_class,
                max_samples=self.max_samples,
                random_state=estimator_random_state,
            )

            estimator.fit(X, y, sample_weight=sample_weights)
            predictions = estimator.predict(X)
            incorrect = predictions != y

            weighted_error = float(np.dot(sample_weights, incorrect))
            weighted_error = np.clip(weighted_error, 0.0, 1.0)

            if weighted_error >= threshold:
                if n_classes == 2:
                    self.stop_reason_ = "weighted_error_greater_than_or_equal_to_0.5"
                else:
                    self.stop_reason_ = "weighted_error_too_high_for_samme"
                self.stopped_early_ = True
                break

            if n_classes == 2:
                alpha = self.learning_rate * 0.5 * np.log(
                    (1.0 - weighted_error) / max(weighted_error, np.finfo(float).eps)
                )
            else:
                alpha = self.learning_rate * (
                    np.log((1.0 - weighted_error) / max(weighted_error, np.finfo(float).eps))
                    + np.log(n_classes - 1)
                )

            self.estimators_.append(estimator)
            self.estimator_errors_.append(weighted_error)
            self.estimator_weights_.append(alpha)

            if weighted_error == 0:
                self.sample_weight_history_.append(sample_weights.copy())
                self.sample_weights_ = sample_weights.copy()
                self.converged_ = True
                self.stopped_early_ = True
                self.stop_reason_ = "perfect_training_accuracy"
                break

            if n_classes == 2:
                update = np.where(
                    incorrect,
                    np.exp(alpha),
                    np.exp(-alpha),
                )
            else:
                update = np.exp(alpha * incorrect.astype(float))
            sample_weights = sample_weights * update
            sample_weights /= np.sum(sample_weights)
            self.sample_weight_history_.append(sample_weights.copy())

        self.sample_weights_ = sample_weights.copy()
        self.n_estimators_ = len(self.estimators_)
        if self.n_estimators_ > 0 and not self.stopped_early_:
            self.converged_ = True
            self.stop_reason_ = "n_estimators_reached"
        return self

    def decision_function(self, X):
        X = self._check_fitted(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes), dtype=float)

        for estimator, alpha in zip(self.estimators_, self.estimator_weights_):
            predictions = estimator.predict(X)
            for class_index, class_label in enumerate(self.classes_):
                scores[:, class_index] += alpha * (predictions == class_label)
        return scores

    def predict(self, X):
        X = self._check_fitted(X)
        if len(self.estimators_) == 0:
            raise ValueError("AdaBoostSvm has no trained estimators")
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _check_fitted(self, X):
        if not self.estimators_:
            raise ValueError("Model has not been trained")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError("Invalid feature dimensions")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains invalid values")
        return X