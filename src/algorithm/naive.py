import numpy as np


class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    # learning
    def fit(self, X, y):
        X = np.asarray(X, dtype=int)
        y = np.asarray(y)

        self.class_labels_ = np.unique(y)
        n_classes = len(self.class_labels_)
        n_features = X.shape[1]

        self.n_categories_ = X.max(axis=0) + 1
        max_categories = int(self.n_categories_.max())

        # Frekuensi per kelas jd prior P(vj)
        self.class_freq_ = np.array([(y == c).sum() for c in self.class_labels_])
        self.log_class_prob_ = np.log(self.class_freq_ / len(y))

        self.feature_log_ = np.full((n_features, n_classes, max_categories), -np.inf)
        self.log_unknown_ = np.zeros((n_features, n_classes))

        with np.errstate(divide='ignore'):
            for feat_idx in range(n_features):
                k = int(self.n_categories_[feat_idx])
                
                for class_idx, label in enumerate(self.class_labels_):
                    freq = np.bincount(X[y == label, feat_idx], minlength=k)
                    denominator = self.class_freq_[class_idx] + self.alpha * k
                    self.feature_log_[feat_idx, class_idx, :k] = np.log((freq + self.alpha) / denominator)
                    self.log_unknown_[feat_idx, class_idx] = np.log(self.alpha / denominator)
                    
        return self

    # Klasifikasi
    def _class_log_score(self, X):
        X = np.asarray(X, dtype=int)
        n_rows = X.shape[0]
        log_scores = np.tile(self.log_class_prob_, (n_rows, 1))

        for feat_idx in range(X.shape[1]):
            x = X[:, feat_idx]
            known = (x >= 0) & (x < self.n_categories_[feat_idx])
            feature_score = np.empty((n_rows, len(self.class_labels_)))
            feature_score[known] = self.feature_log_[feat_idx][:, x[known]].T
            feature_score[~known] = self.log_unknown_[feat_idx]
            log_scores += feature_score
            
        return log_scores

    def predict_proba(self, X):
        log_scores = self._class_log_score(X)
        log_scores = log_scores - log_scores.max(axis=1, keepdims=True)     
        prob = np.exp(log_scores)
        
        return prob / prob.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.class_labels_[np.argmax(self._class_log_score(X), axis=1)]

    def save(self, path):
        np.savez(
            path,
            alpha=self.alpha,
            class_labels=self.class_labels_,
            n_categories=self.n_categories_,
            class_freq=self.class_freq_,
            log_class_prob=self.log_class_prob_,
            feature_log=self.feature_log_,
            log_unknown=self.log_unknown_,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(alpha=float(data['alpha']))
            model.class_labels_ = data['class_labels']
            model.n_categories_ = data['n_categories']
            model.class_freq_ = data['class_freq']
            model.log_class_prob_ = data['log_class_prob']
            model.feature_log_ = data['feature_log']
            model.log_unknown_ = data['log_unknown']

        return model