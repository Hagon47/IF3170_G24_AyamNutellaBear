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
        return self.lmda * (self.l1_ratio * l1 + (1 - self.l1_ratio) * l2)
    
    def regularization_gradient(self):
        # elemen pertama tidak kena penalti
        m = np.zeros_like(self.b)
        w = self.b[1:]
        if self.penalty is None or self.lmda == 0 or self.penalty.lower() == 'none':
            return m

        if self.penalty.lower() == 'l1':
            m[1:] = self.lmda * np.sign(w)
        elif self.penalty.lower() == 'l2':
            m[1:] = self.lmda * w
        else:
            m[1:] = self.lmda * (self.l1_ratio * np.sign(w) + (1 - self.l1_ratio) * w)
        return m
    
    def loss(self, z, y):
        log_loss = np.mean(np.logaddexp(0, (1 - 2 * y) * z))
        return float(log_loss + self.regularization_loss())
    
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X harus matriks 2D yang tidak kosong.")
        if not np.isfinite(X).all():
            raise ValueError("X tidak boleh berisi NaN atau infinity.")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y harus vektor 1D dengan panjang yang sama dengan jumlah baris X.")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Label y harus 0 atau 1.")
        
        X_bias = np.column_stack((np.ones(len(X)), X))
        self.b = np.zeros(X_bias.shape[1])
        self.n_iteration = 0
        self.loss_history = [self.loss(X_bias @ self.b, y)]
        rng = np.random.default_rng(self.random)
        
        for t in range (self.max_iteration):
            order = rng.permutation(len(X)) if self.shuffle else range(len(X))
            for i in order:
                xi = X_bias[i]
                yi = y[i]
                z = xi @ self.b
                pi = self.sigmoid(z)
                grad = (yi - pi) * xi - self.regularization_gradient()
                self.b += self.learning_rate * grad
        
            loss = self.loss(X_bias @ self.b, y)
            if not np.isfinite(loss) or not np.isfinite(self.b).all():
                self.b = None 
                raise ValueError("Loss atau koefisien tidak finite. Coba kurangi learning rate.")
            
            self.loss_history.append(loss)
            self.n_iteration += 1
        return self
    
    def z_score(self, X):
        self.is_fitted()
        X = np.asarray(X, dtype=float)
        
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X harus matriks 2D yang tidak kosong.")
        if not np.isfinite(X).all():
            raise ValueError("X tidak boleh berisi NaN atau infinity.")
        if X.shape[1] != len(self.b) - 1:
            raise ValueError("Jumlah fitur berbeda dengan data training.")
        return self.b[0] + X @ self.b[1:]
    
    def probability(self, X):
        z = self.z_score(X)
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold harus antara 0 dan 1.")
        prob = self.probability(X)
        return (prob >= threshold).astype(int)  
    
    def save(self, path):
        self.is_fitted()
        path = Path(path)
        with path.open("wb") as file:
            np.savez_compressed(
                file,
                b=self.b,
                loss_history=self.loss_history,
                n_iteration=self.n_iteration,
                learning_rate=self.learning_rate,
                max_iteration=self.max_iteration,
                penalty=self.penalty if self.penalty is not None else "none",
                lmda=self.lmda,
                l1_ratio=self.l1_ratio,
                random=self.random,
                shuffle=self.shuffle,
            )
    
    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            penalty = data["penalty"].item()
            model = cls(
                learning_rate=data["learning_rate"].item(),
                max_iteration=data["max_iteration"].item(),
                penalty=None if penalty == "none" else penalty,
                lmda=data["lmda"].item(),
                l1_ratio=data["l1_ratio"].item(),
                random=data["random"].item(),
                shuffle=data["shuffle"].item(),
            )
            model.b = data["b"]
            model.loss_history = data["loss_history"].tolist()
            model.n_iteration = data["n_iteration"].item()

        return model