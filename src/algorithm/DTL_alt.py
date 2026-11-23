import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Struktur penyimpan informasi setiap simpul (node) pada pohon
class Node:
    def __init__(
        self,
        feature_idx=None,
        feature_name=None,
        is_continuous=False,
        threshold=None,
        children=None,
        branch_weights=None,
        is_leaf=False,
        prediction=None,
        class_probs=None,
        entropy=0.0,
        gain_ratio=0.0,
        n_samples=0.0,
        n_errors=0.0,
    ):
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.is_continuous = is_continuous
        self.threshold = threshold
        self.children = children or {}
        self.branch_weights = branch_weights or {}
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.class_probs = class_probs
        self.entropy = entropy
        self.gain_ratio = gain_ratio
        self.n_samples = n_samples
        self.n_errors = n_errors

# Implementasi Decision Tree Learning (C4.5) yang mendukung numerik, kategorikal, dan missing value
# L1. Hitung Entropi awal node saat ini (atau aproksimasi L'Hospital jika criterion='lhospital')
# L2. Tentukan kategori atribut saat ini: kontinu, kategorikal, mengandung missing value
# L3. Hitung Gain berdasarkan kategori atribut (binary threshold atau MSD-Splitting 4-bin)
# L4. Hitung Split Info dan Gain Ratio
# L5. Opsional: lakukan Post-Pruning (PEP / REP) setelah pohon selesai dibangun

# Menambahkan 4 optimasi:
# 1. Cost-Sensitive Learning (class_weight: None / 'balanced' / dict)
# 2. Threshold Tuning pada predict(X, threshold=0.5)
# 3. MSD-Splitting 4-bin untuk atribut kontinu (continuous_split='msd')
# 4. Aproksimasi L'Hospital tanpa logaritma (criterion='lhospital')

class DTL_Alt:
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_gain=1e-6,
        categorical_features=None,
        pruning=None,
        val_ratio=0.15,
        rep_metric='macro_f1',
        random_state=42,
        class_weight=None,
        continuous_split='binary',
        criterion='gain_ratio',
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.categorical_features = categorical_features
        self.pruning = pruning
        self.val_ratio = float(val_ratio)
        self.rep_metric = rep_metric
        self.random_state = random_state

        self.class_weight = class_weight
        if continuous_split not in ('binary', 'msd'):
            raise ValueError("continuous_split harus bernilai 'binary' atau 'msd'.")
        self.continuous_split = continuous_split
        if criterion not in ('gain_ratio', 'lhospital'):
            raise ValueError("criterion harus bernilai 'gain_ratio' atau 'lhospital'.")
        self.criterion = criterion

        self.root = None
        self.classes = None
        self.feature_names = None
        self.is_continuous = None

    # L2 Mendeteksi baris yang bernilai kosong (NaN/None) pada kolom angka maupun string
    def is_missing(self, col):
        if col.dtype.kind in ('i', 'u', 'b'):
            return np.zeros(len(col), dtype=bool)
        if col.dtype.kind in ('f', 'c'):
            return np.isnan(col)
        mask = np.zeros(len(col), dtype=bool)
        for i, val in enumerate(col):
            if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)) or str(val) == 'nan':
                mask[i] = True
        return mask

    # L2 Menentukan kategori tiap atribut: True untuk kontinu, False untuk kategorikal (termasuk string)
    def detect_feature_types(self, X):
        n_features = X.shape[1]
        is_cont = np.zeros(n_features, dtype=bool)
        for j in range(n_features):
            if self.categorical_features is not None:
                if j in self.categorical_features or (
                    self.feature_names and self.feature_names[j] in self.categorical_features
                ):
                    is_cont[j] = False
                    continue

            col = X[:, j]
            not_nan = col[~self.is_missing(col)]
            if len(not_nan) == 0:
                is_cont[j] = False
                continue

            try:
                not_nan_float = not_nan.astype(float)
                if np.unique(not_nan_float).size > 15 or not np.all(np.mod(not_nan_float, 1) == 0):
                    is_cont[j] = True
            except (ValueError, TypeError):
                # Gagal dikonversi ke float menandakan atribut bertipe string (kategorikal)
                is_cont[j] = False
        return is_cont

    # Method baru untuk menghitung bobot awal tiap baris berdasarkan class_weight
    def compute_sample_weights(self, y_encoded):
        n_samples = len(y_encoded)
        if self.class_weight is None:
            return np.ones(n_samples, dtype=float)

        n_classes = len(self.classes)
        if self.class_weight == 'balanced':
            counts = np.bincount(y_encoded, minlength=n_classes)
            cw = np.where(counts > 0, n_samples / (n_classes * counts), 0.0)
            return cw[y_encoded].astype(float)

        if isinstance(self.class_weight, dict):
            cw = np.array([float(self.class_weight.get(cls, 1.0)) for cls in self.classes], dtype=float)
            return cw[y_encoded]

        raise ValueError("class_weight harus bernilai None, 'balanced', atau dict.")

    # L1, L3 Menghitung nilai Entropi E(S) = -sum(p_i * log2(p_i))
    # Jika criterion='lhospital', mengganti -sum(p*log2(p)) menjadi 1 - sum(p^2) tanpa operasi log
    def entropy(self, y, weights):
        total_w = np.sum(weights)
        if total_w <= 0:
            return 0.0
        class_counts = np.bincount(y, weights=weights, minlength=len(self.classes))
        probs = class_counts / total_w
        if self.criterion == 'lhospital':
            return float(1.0 - np.sum(probs * probs))
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    # L1, L3 Menghitung proporsi distribusi kelas pada node saat ini untuk menentukan prediksi mayoritas
    def class_probs(self, y, weights):
        total_w = np.sum(weights)
        if total_w <= 0:
            return np.ones(len(self.classes)) / len(self.classes)
        class_counts = np.bincount(y, weights=weights, minlength=len(self.classes))
        return class_counts / total_w

    # L2, L3 Diskritisasi atribut kontinu untuk mencari titik potong (threshold) dengan Gain tertinggi
    def evaluate_continuous(self, col_known, y_known, w_known, e_known):
        # Urutkan data kontinu dari nilai terkecil ke terbesar
        vals = col_known.astype(float)
        order = np.argsort(vals)
        vals_sorted = vals[order]
        y_sorted = y_known[order]
        w_sorted = w_known[order]

        # Cari posisi perubahan nilai sebagai kandidat titik potong
        diff_mask = vals_sorted[:-1] != vals_sorted[1:]
        if not np.any(diff_mask):
            return -1.0, None, None

        # Akumulasi bobot tiap kelas untuk cabang kiri (<= t) dan kanan (> t) secara vektorisasi
        n_classes = len(self.classes)
        one_hot_w = np.zeros((len(y_sorted), n_classes), dtype=float)
        one_hot_w[np.arange(len(y_sorted)), y_sorted] = w_sorted

        left_class_w = np.cumsum(one_hot_w, axis=0)[:-1][diff_mask]
        total_class_w = one_hot_w.sum(axis=0)
        right_class_w = total_class_w - left_class_w

        left_w = left_class_w.sum(axis=1, keepdims=True)
        right_w = right_class_w.sum(axis=1, keepdims=True)
        total_w = float(np.sum(w_known))

        valid_mask = (left_w.ravel() >= self.min_samples_leaf) & (right_w.ravel() >= self.min_samples_leaf)
        if not np.any(valid_mask):
            return -1.0, None, None

        # Hitung Entropi cabang kiri, cabang kanan, dan Gain untuk seluruh kandidat titik potong
        # Pada mode 'lhospital', impuritas cabang dihitung dengan 1 - sum(p^2) tanpa np.log2
        with np.errstate(divide='ignore', invalid='ignore'):
            p_left = np.where(left_w > 0, left_class_w / left_w, 0.0)
            p_right = np.where(right_w > 0, right_class_w / right_w, 0.0)
            if self.criterion == 'lhospital':
                e_left = 1.0 - np.sum(p_left * p_left, axis=1)
                e_right = 1.0 - np.sum(p_right * p_right, axis=1)
            else:
                e_left = -np.sum(np.where(p_left > 0, p_left * np.log2(p_left), 0.0), axis=1)
                e_right = -np.sum(np.where(p_right > 0, p_right * np.log2(p_right), 0.0), axis=1)

        p_left_branch = left_w.ravel() / total_w
        p_right_branch = right_w.ravel() / total_w
        e_children = p_left_branch * e_left + p_right_branch * e_right
        gains = np.where(valid_mask, e_known - e_children, -np.inf)

        # Pilih titik potong (nilai tengah) yang menghasilkan Gain terbesar
        best_idx = int(np.argmax(gains))
        best_gain = float(gains[best_idx])
        thresholds = (vals_sorted[:-1][diff_mask] + vals_sorted[1:][diff_mask]) / 2.0
        best_threshold = float(thresholds[best_idx])

        branch_w = (float(left_w.ravel()[best_idx]), float(right_w.ravel()[best_idx]))
        return best_gain, branch_w, best_threshold

    # Method baru untuk diskritisasi atribut kontinu menjadi 4 bin: (mu-sigma, mu, mu+sigma)
    def evaluate_continuous_msd(self, col_known, y_known, w_known, e_known):
        vals = col_known.astype(float)
        total_w = float(np.sum(w_known))
        if total_w <= 0:
            return -1.0, None, None

        # Hitung rata-rata (mu) dan simpangan baku (sigma) berbobot agar konsisten dengan bobot C4.5 / class_weight
        mu = float(np.sum(w_known * vals) / total_w)
        var = float(np.sum(w_known * (vals - mu) ** 2) / total_w)
        sigma = float(np.sqrt(max(var, 0.0)))
        if sigma <= 1e-12:
            return -1.0, None, None

        # 4 bin MSD-Splitting: bin 0 (< mu-sigma), bin 1 ([mu-sigma, mu)), bin 2 ([mu, mu+sigma)), bin 3 (>= mu+sigma)
        cutpoints = (mu - sigma, mu, mu + sigma)
        bin_idx = np.digitize(vals, cutpoints, right=False)

        e_children = 0.0
        branch_w = []
        valid_branches = 0

        for b in range(4):
            mask_b = bin_idx == b
            wb = w_known[mask_b]
            sum_wb = float(np.sum(wb))
            if sum_wb >= self.min_samples_leaf:
                valid_branches += 1
            branch_w.append(sum_wb)
            if sum_wb > 0:
                p_branch = sum_wb / total_w
                e_children += p_branch * self.entropy(y_known[mask_b], wb)

        # Minimal 2 bin harus memenuhi min_samples_leaf agar pemisahan 4-bin bermakna
        if valid_branches < 2:
            return -1.0, None, None

        raw_gain = e_known - e_children
        return float(raw_gain), branch_w, cutpoints

    # L2, L3 Menghitung Entropi percabangan dan Gain untuk atribut kategorikal (multi-way split)
    def evaluate_categorical(self, col_known, y_known, w_known, e_known):
        unique_vals, inv_idx = np.unique(col_known, return_inverse=True)
        if len(unique_vals) <= 1:
            return -1.0, None, None

        total_w = float(np.sum(w_known))
        e_children = 0.0
        branch_w = []
        valid_branches = 0

        # Hitung Entropi tertimbang untuk setiap kategori unik
        for k in range(len(unique_vals)):
            mask_k = inv_idx == k
            wk = w_known[mask_k]
            sum_wk = float(np.sum(wk))
            if sum_wk >= self.min_samples_leaf:
                valid_branches += 1
            branch_w.append(sum_wk)
            if sum_wk > 0:
                p_branch = sum_wk / total_w
                e_children += p_branch * self.entropy(y_known[mask_k], wk)

        if valid_branches < 2:
            return -1.0, None, None
        raw_gain = e_known - e_children
        return float(raw_gain), branch_w, unique_vals

    # L3 Menghitung SplitInfo = -sum(p_part * log2(p_part)) sebagai faktor penalti jumlah cabang
    # Jika criterion='lhospital', mengganti SplitInfo log2 menjadi Split-Gini 1 - sum(p_part^2)
    def calculate_split_info(self, part_weights, total_w):
        if self.criterion == 'lhospital':
            sum_sq = 0.0
            for part_w in part_weights:
                if part_w > 0:
                    p_part = part_w / total_w
                    sum_sq += p_part * p_part
            return float(1.0 - sum_sq)

        split_info = 0.0
        for part_w in part_weights:
            if part_w > 0:
                p_part = part_w / total_w
                split_info -= p_part * np.log2(p_part)
        return float(split_info)

    # L3 Menghitung Gain Ratio = Gain / SplitInfo untuk seluruh kandidat fitur
    def calculate_gain_ratio(self, gain, split_info):
        if split_info <= 1e-12:
            return 0.0
        return float(gain / split_info)

    # Mengevaluasi seluruh atribut yang tersedia untuk memilih 1 pemisah terbaik berdasarkan Gain Ratio
    def find_best_split(self, X, y, weights, available_features):
        total_w = float(np.sum(weights))
        min_req = max(self.min_samples_split, 2 * self.min_samples_leaf)
        candidates = []

        for j in available_features:
            # 1. Pisahkan data lengkap dan missing value, lalu hitung proporsi data terisi (p_known)
            col = X[:, j]
            miss_mask = self.is_missing(col)
            known_mask = ~miss_mask

            w_known = weights[known_mask]
            sum_w_known = float(np.sum(w_known))
            if sum_w_known < min_req:
                continue

            p_known = sum_w_known / total_w
            y_known = y[known_mask]
            col_known = col[known_mask]

            # 2. Hitung Entropi awal khusus (e_known) hanya dari baris yang tidak missing
            e_known = self.entropy(y_known, w_known)

            # 3. Evaluasi atribut sesuai kategorinya
            # Arahkan atribut kontinu ke evaluate_continuous_msd jika continuous_split='msd'
            if self.is_continuous[j]:
                if self.continuous_split == 'msd':
                    raw_gain, branch_w, threshold = self.evaluate_continuous_msd(col_known, y_known, w_known, e_known)
                else:
                    raw_gain, branch_w, threshold = self.evaluate_continuous(col_known, y_known, w_known, e_known)
                if raw_gain <= 0 or threshold is None:
                    continue
            else:
                raw_gain, branch_w, _ = self.evaluate_categorical(col_known, y_known, w_known, e_known)
                if raw_gain <= 0 or branch_w is None:
                    continue
                threshold = None

            # 4. Koreksi Gain dengan proporsi data terisi (jika bersih tanpa NaN, p_known = 1.0)
            gain = p_known * raw_gain
            if gain < self.min_gain:
                continue

            # 5. Hitung SplitInfo (bila ada missing value, dimasukkan sebagai partisi tambahan) dan Gain Ratio
            w_miss = float(np.sum(weights[miss_mask]))
            all_parts = list(branch_w) + ([w_miss] if w_miss > 0 else [])
            split_info = self.calculate_split_info(all_parts, total_w)
            gain_ratio = self.calculate_gain_ratio(gain, split_info)

            candidates.append({
                'feature_idx': j,
                'gain': gain,
                'gain_ratio': gain_ratio,
                'threshold': threshold,
            })

        if not candidates:
            return None

        # 6. Saring kandidat dengan Gain >= rata-rata Gain (aturan Quinlan), lalu pilih Gain Ratio terbesar
        avg_gain = float(np.mean([c['gain'] for c in candidates]))
        valid_candidates = [c for c in candidates if c['gain'] >= avg_gain - 1e-12]
        best = max(valid_candidates, key=lambda c: (c['gain_ratio'], c['gain']))
        return best

    # Membangun pohon secara rekursif, mewariskan bobot missing value ke cabang anak
    def build_tree(self, X, y, weights, available_features, depth):
        # Hitung Entropi awal dan distribusi kelas pada node saat ini
        total_w = float(np.sum(weights))
        probs = self.class_probs(y, weights)
        majority_idx = int(np.argmax(probs))
        majority_class = self.classes[majority_idx]
        e_current = self.entropy(y, weights)

        # Hitung bobot kesalahan jika node ini dijadikan leaf (digunakan oleh PEP)
        class_counts = np.bincount(y, weights=weights, minlength=len(self.classes))
        n_errors = float(max(0.0, total_w - class_counts[majority_idx]))

        # Cek kondisi berhenti (daun sudah murni, data terlalu sedikit, fitur habis, atau mencapai max_depth)
        if (
            e_current <= 1e-12
            or total_w < self.min_samples_split
            or total_w < 2 * self.min_samples_leaf
            or len(available_features) == 0
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return Node(
                is_leaf=True,
                prediction=majority_class,
                class_probs=probs,
                entropy=e_current,
                n_samples=total_w,
                n_errors=n_errors,
            )

        # Cari atribut pemisah terbaik untuk node saat ini
        best_split = self.find_best_split(X, y, weights, available_features)
        if best_split is None:
            return Node(
                is_leaf=True,
                prediction=majority_class,
                class_probs=probs,
                entropy=e_current,
                n_samples=total_w,
                n_errors=n_errors,
            )

        j = best_split['feature_idx']
        threshold = best_split['threshold']
        is_cont = bool(self.is_continuous[j])

        node = Node(
            feature_idx=j,
            feature_name=self.feature_names[j],
            is_continuous=is_cont,
            threshold=threshold,
            prediction=majority_class,
            class_probs=probs,
            entropy=e_current,
            gain_ratio=best_split['gain_ratio'],
            n_samples=total_w,
            n_errors=n_errors,
        )

        # Kelompokkan indeks baris yang nilainya diketahui ke masing-masing cabang
        col = X[:, j]
        miss_mask = self.is_missing(col)
        known_mask = ~miss_mask
        sum_w_known = float(np.sum(weights[known_mask]))
        known_indices = np.where(known_mask)[0]
        miss_indices = np.where(miss_mask)[0]

        # Atribut kategorikal yang sudah dipakai dikeluarkan dari kandidat cabang di bawahnya
        next_features = available_features if is_cont else [f for f in available_features if f != j]

        # Jika threshold berupa tuple 3 titik potong, bentuk 4 cabang bin (0, 1, 2, 3)
        if is_cont:
            vals_known = col[known_mask].astype(float)
            if isinstance(threshold, tuple):
                bin_idx = np.digitize(vals_known, threshold, right=False)
                branches = {b: known_indices[bin_idx == b] for b in range(4)}
            else:
                branches = {
                    '<=': known_indices[vals_known <= threshold],
                    '>': known_indices[vals_known > threshold],
                }
        else:
            unique_vals = np.unique(col[known_mask])
            branches = {
                val: known_indices[col[known_mask] == val]
                for val in unique_vals
            }

        # Wariskan bobot proporsional (p_branch) untuk baris missing value dan bangun tiap cabang secara rekursif
        for branch_key, idx_known in branches.items():
            w_branch = float(np.sum(weights[idx_known]))
            p_branch = w_branch / sum_w_known if sum_w_known > 0 else 0.0
            node.branch_weights[branch_key] = p_branch

            if len(idx_known) == 0:
                node.children[branch_key] = Node(
                    is_leaf=True,
                    prediction=majority_class,
                    class_probs=probs,
                    entropy=0.0,
                    n_samples=0.0,
                    n_errors=0.0,
                )
                continue

            if len(miss_indices) > 0 and p_branch > 0:
                sub_idx = np.concatenate([idx_known, miss_indices])
                sub_weights = np.concatenate([weights[idx_known], weights[miss_indices] * p_branch])
            else:
                sub_idx = idx_known
                sub_weights = weights[idx_known]

            node.children[branch_key] = self.build_tree(
                X[sub_idx], y[sub_idx], sub_weights, next_features, depth + 1
            )

        return node

    # L5 Mengubah internal node menjadi leaf saat subtree dipangkas
    def collapse_to_leaf(self, node):
        node.is_leaf = True
        node.children = {}
        node.branch_weights = {}
        node.feature_idx = None
        node.feature_name = None
        node.threshold = None
        node.gain_ratio = 0.0

    # L5 Menghitung total error dan jumlah daun pada subtree untuk Pessimistic Error Pruning (PEP)
    def subtree_pep_stats(self, node):
        if node.is_leaf:
            return node.n_errors, (1 if node.n_samples > 0 else 0)
        total_err = 0.0
        total_leaves = 0
        for child in node.children.values():
            c_err, c_leaves = self.subtree_pep_stats(child)
            total_err += c_err
            total_leaves += c_leaves
        return total_err, max(1, total_leaves)

    # L5 Pessimistic Error Pruning (PEP) secara top-down tanpa memotong data latih
    def prune_pep(self, node):
        if node.is_leaf:
            return
        n_total = node.n_samples
        if n_total <= 0:
            self.collapse_to_leaf(node)
            return

        e_sub, n_leaves = self.subtree_pep_stats(node)
        e_prime_leaf = node.n_errors + 0.5
        e_prime_sub = e_sub + 0.5 * n_leaves
        p_sub = min(max(e_prime_sub / n_total, 0.0), 1.0)
        se_sub = float(np.sqrt(n_total * p_sub * (1.0 - p_sub)))

        if e_prime_leaf <= e_prime_sub + se_sub:
            self.collapse_to_leaf(node)
            return

        for child in node.children.values():
            self.prune_pep(child)

    # L5 Membagi data secara stratified di dalam fit() untuk validation set internal REP
    def stratified_internal_split(self, y_encoded):
        rng = np.random.default_rng(self.random_state)
        train_idx = []
        val_idx = []
        for cls_idx in range(len(self.classes)):
            cls_pos = np.flatnonzero(y_encoded == cls_idx)
            rng.shuffle(cls_pos)
            n_val = int(round(len(cls_pos) * self.val_ratio))
            if len(cls_pos) >= 2:
                n_val = max(1, min(n_val, len(cls_pos) - 1))
            else:
                n_val = 0
            val_idx.append(cls_pos[:n_val])
            train_idx.append(cls_pos[n_val:])
        train_idx = np.concatenate(train_idx)
        val_idx = np.concatenate(val_idx)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        return train_idx, val_idx

    # L5 Menghitung skor evaluasi REP ('macro_f1' atau 'accuracy') langsung dari confusion matrix
    def score_from_cm(self, cm):
        total = cm.sum()
        if total == 0:
            return 0.0
        if self.rep_metric == 'accuracy':
            return float(np.trace(cm) / total)
        if self.rep_metric == 'macro_f1':
            tp = np.diag(cm).astype(float)
            fp = cm.sum(axis=0) - tp
            fn = cm.sum(axis=1) - tp
            denom = 2.0 * tp + fp + fn
            f1s = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
            return float(np.mean(f1s))
        raise ValueError("rep_metric harus bernilai 'macro_f1' atau 'accuracy'.")

    # L5 Reduced Error Pruning (REP) secara bottom-up dengan pembaruan confusion matrix inkremental
    def prune_rep(self, X_val, y_val_encoded):
        n_val = len(y_val_encoded)
        if n_val == 0 or self.root is None or self.root.is_leaf:
            return

        n_classes = len(self.classes)
        current_probs = self.predict_proba(X_val)
        current_preds = np.argmax(current_probs, axis=1)

        cm = np.zeros((n_classes, n_classes), dtype=int)
        np.add.at(cm, (y_val_encoded, current_preds), 1)
        state = {'score': self.score_from_cm(cm)}

        def recurse(node, idx, route_w):
            if len(idx) == 0:
                if not node.is_leaf:
                    # Pangkas cabang yang tidak pernah dilewati data validasi REP
                    self.collapse_to_leaf(node)
                return np.empty((0, n_classes), dtype=float)

            if node.is_leaf:
                return np.broadcast_to(node.class_probs, (len(idx), n_classes)).copy()

            col = X_val[idx, node.feature_idx]
            miss_mask = self.is_missing(col)
            known_mask = ~miss_mask

            sub_probs = np.zeros((len(idx), n_classes), dtype=float)
            has_branch = np.zeros(len(idx), dtype=bool)

            # Mendukung pengarahan baris validasi REP ke 4 bin MSD maupun 2 cabang binary
            if node.is_continuous:
                known_pos = np.flatnonzero(known_mask)
                if isinstance(node.threshold, tuple):
                    if len(known_pos) > 0:
                        vals_known = col[known_pos].astype(float)
                        bin_idx = np.digitize(vals_known, node.threshold, right=False)
                        branch_positions = {b: known_pos[bin_idx == b] for b in range(4)}
                    else:
                        branch_positions = {b: np.array([], dtype=int) for b in range(4)}
                else:
                    if len(known_pos) > 0:
                        vals_known = col[known_pos].astype(float)
                        branch_positions = {
                            '<=': known_pos[vals_known <= node.threshold],
                            '>': known_pos[vals_known > node.threshold],
                        }
                    else:
                        branch_positions = {'<=': np.array([], dtype=int), '>': np.array([], dtype=int)}
            else:
                branch_positions = {}
                known_pos = np.flatnonzero(known_mask)
                if len(known_pos) > 0:
                    vals_known = col[known_pos]
                    for b_key in node.children:
                        branch_positions[b_key] = known_pos[vals_known == b_key]

            miss_pos = np.flatnonzero(miss_mask)

            for b_key, child in node.children.items():
                k_pos = branch_positions.get(b_key, np.array([], dtype=int))
                p_branch = node.branch_weights.get(b_key, 0.0)

                if len(miss_pos) > 0 and p_branch > 0:
                    c_pos = np.concatenate([k_pos, miss_pos])
                    c_idx = idx[c_pos]
                    c_rw = np.concatenate([route_w[k_pos], route_w[miss_pos] * p_branch])
                    c_probs = recurse(child, c_idx, c_rw)
                    if len(k_pos) > 0:
                        sub_probs[k_pos] = c_probs[:len(k_pos)]
                        has_branch[k_pos] = True
                    sub_probs[miss_pos] += p_branch * c_probs[len(k_pos):]
                    has_branch[miss_pos] = True
                else:
                    c_idx = idx[k_pos]
                    c_rw = route_w[k_pos]
                    c_probs = recurse(child, c_idx, c_rw)
                    if len(k_pos) > 0:
                        sub_probs[k_pos] = c_probs
                        has_branch[k_pos] = True

            # Fallback ke probabilitas node induk untuk kategori asing atau missing tanpa cabang berbobot positif
            fallback_mask = (~has_branch) | (miss_mask & (sub_probs.sum(axis=1) <= 0))
            if np.any(fallback_mask):
                sub_probs[fallback_mask] = node.class_probs

            # Evaluasi dampak jika node ini dipangkas menjadi leaf
            delta = route_w[:, None] * (node.class_probs[None, :] - sub_probs)
            cand_probs = current_probs[idx] + delta
            cand_preds = np.argmax(cand_probs, axis=1)

            old_preds = current_preds[idx]
            changed = cand_preds != old_preds

            if np.any(changed):
                ch_y = y_val_encoded[idx[changed]]
                ch_old = old_preds[changed]
                ch_new = cand_preds[changed]
                cm_cand = cm.copy()
                np.add.at(cm_cand, (ch_y, ch_old), -1)
                np.add.at(cm_cand, (ch_y, ch_new), 1)
                cand_score = self.score_from_cm(cm_cand)
            else:
                cm_cand = cm
                cand_score = state['score']

            # Pangkas jika skor validasi meningkat atau tetap sama (memilih pohon lebih sederhana)
            if cand_score >= state['score'] - 1e-12:
                self.collapse_to_leaf(node)
                current_probs[idx] = cand_probs
                if np.any(changed):
                    current_preds[idx] = cand_preds
                    cm[:, :] = cm_cand
                    state['score'] = cand_score
                return np.broadcast_to(node.class_probs, (len(idx), n_classes)).copy()

            return sub_probs

        all_idx = np.arange(n_val, dtype=int)
        all_rw = np.ones(n_val, dtype=float)
        recurse(self.root, all_idx, all_rw)

    # Entry point pelatihan model: siapkan array NumPy, tipe fitur, dan bobot awal lalu bangun pohon
    def fit(self, X, y, feature_names=None):
        if hasattr(X, 'columns') and hasattr(X, 'to_numpy'):
            self.feature_names = list(X.columns)
            X_arr = X.to_numpy(dtype=object)
        else:
            X_arr = np.asarray(X)
            self.feature_names = feature_names or [f'Feature_{i}' for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y)
        self.classes, y_encoded = np.unique(y_arr, return_inverse=True)
        self.is_continuous = self.detect_feature_types(X_arr)
        available_features = list(range(X_arr.shape[1]))

        if self.pruning == 'rep':
            fit_idx, val_idx = self.stratified_internal_split(y_encoded)
            X_fit, y_fit = X_arr[fit_idx], y_encoded[fit_idx]
            X_rep, y_rep = X_arr[val_idx], y_encoded[val_idx]
            # Menggunakan compute_sample_weights() sebagai pengganti np.ones()
            weights = self.compute_sample_weights(y_fit)
            self.root = self.build_tree(X_fit, y_fit, weights, available_features, depth=0)
            self.prune_rep(X_rep, y_rep)
        elif self.pruning in (None, 'pep'):
            # Menggunakan compute_sample_weights() sebagai pengganti np.ones()
            weights = self.compute_sample_weights(y_encoded)
            self.root = self.build_tree(X_arr, y_encoded, weights, available_features, depth=0)
            if self.pruning == 'pep':
                self.prune_pep(self.root)
        else:
            raise ValueError("pruning harus bernilai None, 'pep', atau 'rep'.")

        return self

    # Menelusuri pohon untuk satu baris data uji (menangani missing value dan kategori asing)
    def predict_row_probs(self, row, node):
        if node.is_leaf:
            return node.class_probs

        val = row[node.feature_idx]
        is_nan = val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)) or str(val) == 'nan'

        # Jika nilai pada data uji kosong, gabungkan probabilitas seluruh cabang anak sesuai proporsi bobotnya
        if is_nan:
            combined_probs = np.zeros(len(self.classes), dtype=float)
            for branch_key, child in node.children.items():
                p_branch = node.branch_weights.get(branch_key, 0.0)
                if p_branch > 0:
                    combined_probs += p_branch * self.predict_row_probs(row, child)
            return combined_probs if np.sum(combined_probs) > 0 else node.class_probs

        # Jika node.threshold adalah tuple 3 cutpoint, tentukan indeks bin (0..3)
        if node.is_continuous:
            fval = float(val)
            if isinstance(node.threshold, tuple):
                branch_key = int(np.digitize(fval, node.threshold, right=False))
            else:
                branch_key = '<=' if fval <= node.threshold else '>'
            child = node.children.get(branch_key)
            return self.predict_row_probs(row, child) if child else node.class_probs
        else:
            child = node.children.get(val)
            return self.predict_row_probs(row, child) if child else node.class_probs

    # Menghasilkan matriks probabilitas kelas untuk seluruh baris data uji
    def predict_proba(self, X):
        X_arr = X.to_numpy(dtype=object) if hasattr(X, 'to_numpy') else np.asarray(X)
        return np.array([self.predict_row_probs(row, self.root) for row in X_arr])

    # Menghasilkan prediksi label kelas akhir berdasarkan probabilitas tertinggi
    # Menambahkan parameter threshold untuk menggeser ambang keputusan klasifikasi biner
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        if len(self.classes) == 2 and threshold is not None:
            return np.where(probs[:, 1] >= threshold, self.classes[1], self.classes[0])
        return self.classes[np.argmax(probs, axis=1)]

    # Menyimpan objek model yang sudah dilatih ke berkas
    def save(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    # Memuat kembali objek model dari berkas
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    # Menggambar visualisasi percabangan pohon hingga sebanyak top_n percabangan teratas
    def plot_tree(self, top_n=5, figsize=(16, 10), save_path=None):
        if self.root is None:
            raise ValueError("Model belum dilatih. Panggil fit() terlebih dahulu.")

        # Pilih top-N internal node teratas menggunakan penelusuran level (BFS)
        expanded_nodes = set()
        queue = [self.root]
        while queue and len(expanded_nodes) < top_n:
            curr = queue.pop(0)
            if not curr.is_leaf and len(curr.children) > 0:
                expanded_nodes.add(id(curr))
                for child in curr.children.values():
                    if not child.is_leaf:
                        queue.append(child)

        positions = {}
        edges = []
        leaf_counter = [0]

        def assign_coords(node, depth=0):
            if node.is_leaf or id(node) not in expanded_nodes:
                x = float(leaf_counter[0])
                leaf_counter[0] += 1.0
                positions[id(node)] = (x, -depth)
                return x

            child_xs = []
            for branch_key, child in node.children.items():
                cx = assign_coords(child, depth + 1)
                child_xs.append(cx)
                # Format label interval 4-bin saat node.threshold berupa tuple
                if node.is_continuous:
                    if isinstance(node.threshold, tuple):
                        t1, t2, t3 = node.threshold
                        msd_labels = {
                            0: f"< {t1:.2f}",
                            1: f"[{t1:.2f}, {t2:.2f})",
                            2: f"[{t2:.2f}, {t3:.2f})",
                            3: f">= {t3:.2f}",
                        }
                        label = msd_labels.get(branch_key, f"bin {branch_key}")
                    else:
                        label = f"{branch_key} {node.threshold:.2f}"
                else:
                    label = f"= {branch_key}"
                edges.append((id(node), id(child), label))

            x = float(np.mean(child_xs)) if child_xs else float(leaf_counter[0])
            positions[id(node)] = (x, -depth)
            return x

        assign_coords(self.root, depth=0)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()

        for parent_id, child_id, edge_label in edges:
            x1, y1 = positions[parent_id]
            x2, y2 = positions[child_id]
            ax.plot([x1, x2], [y1, y2], color='#555555', lw=1.5, zorder=1)
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            ax.text(
                mx, my, edge_label, fontsize=8.5, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85), zorder=2
            )

        def draw_nodes(node):
            x, y = positions[id(node)]
            is_shown_internal = (not node.is_leaf) and (id(node) in expanded_nodes)

            if is_shown_internal:
                # Menyesuaikan label impuritas (I vs E) sesuai criterion yang dipakai
                imp_label = "I" if self.criterion == 'lhospital' else "E"
                text = (
                    f"{node.feature_name}\n"
                    f"GainRatio: {node.gain_ratio:.4f} | {imp_label}: {node.entropy:.3f}\n"
                    f"Samples: {node.n_samples:.0f} | Pred: {node.prediction}"
                )
                box_color = '#D6EAF8'
                border_color = '#2E86C1'
            else:
                status = "Leaf" if node.is_leaf else "Truncated"
                imp_label = "I" if self.criterion == 'lhospital' else "E"
                text = (
                    f"[{status}] Pred: {node.prediction}\n"
                    f"{imp_label}: {node.entropy:.3f} | Samples: {node.n_samples:.0f}\n"
                    f"Probs: {np.round(node.class_probs, 2).tolist()}"
                )
                box_color = '#D5F5E3' if node.prediction == 0 else '#FADBD8'
                border_color = '#28B463' if node.prediction == 0 else '#CB4335'

            ax.text(
                x, y, text, fontsize=8.5, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.45', fc=box_color, ec=border_color, lw=1.5),
                zorder=3
            )

            if is_shown_internal:
                for child in node.children.values():
                    draw_nodes(child)

        draw_nodes(self.root)
        # Menampilkan ringkasan mode optimasi aktif pada judul visualisasi
        ax.set_title(
            f"Visualisasi Decision Tree C4.5 Alt ({self.continuous_split}, {self.criterion}, cw={self.class_weight}) - Top-{len(expanded_nodes)}",
            fontsize=12, fontweight='bold', pad=15,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig