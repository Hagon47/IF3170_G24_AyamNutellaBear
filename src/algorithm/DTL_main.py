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

# Implementasi Decision Tree Learning (C4.5) yang mendukung numerik, kategorikal, dan missing value
# L1. Hitung Entropi awal node saat ini
# L2. Tentukan kategori atribut saat ini: kontinu, kategorikal, mengandung missing value
# L3. Hitung Gain berdasarkan kategori atribut
# L4. Hitung Split Info dan Gain Ratio
class DTL_Main:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_gain=1e-6, categorical_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.categorical_features = categorical_features
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

    # L1, L3 Menghitung nilai Entropi E(S) = -sum(p_i * log2(p_i))
    def entropy(self, y, weights):
        total_w = np.sum(weights)
        if total_w <= 0:
            return 0.0
        class_counts = np.bincount(y, weights=weights, minlength=len(self.classes))
        probs = class_counts / total_w
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
        with np.errstate(divide='ignore', invalid='ignore'):
            p_left = np.where(left_w > 0, left_class_w / left_w, 0.0)
            e_left = -np.sum(np.where(p_left > 0, p_left * np.log2(p_left), 0.0), axis=1)

            p_right = np.where(right_w > 0, right_class_w / right_w, 0.0)
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
    def calculate_split_info(self, part_weights, total_w):
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
            if self.is_continuous[j]:
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
        majority_class = self.classes[int(np.argmax(probs))]
        e_current = self.entropy(y, weights)

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

        if is_cont:
            vals_known = col[known_mask].astype(float)
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

        weights = np.ones(len(y_encoded), dtype=float)
        available_features = list(range(X_arr.shape[1]))

        self.root = self.build_tree(X_arr, y_encoded, weights, available_features, depth=0)
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

        if node.is_continuous:
            branch_key = '<=' if float(val) <= node.threshold else '>'
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
    def predict(self, X):
        probs = self.predict_proba(X)
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
                label = f"{branch_key} {node.threshold:.2f}" if node.is_continuous else f"= {branch_key}"
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
                text = (
                    f"{node.feature_name}\n"
                    f"GainRatio: {node.gain_ratio:.4f} | E: {node.entropy:.3f}\n"
                    f"Samples: {node.n_samples:.0f} | Pred: {node.prediction}"
                )
                box_color = '#D6EAF8'
                border_color = '#2E86C1'
            else:
                status = "Leaf" if node.is_leaf else "Truncated"
                text = (
                    f"[{status}] Pred: {node.prediction}\n"
                    f"E: {node.entropy:.3f} | Samples: {node.n_samples:.0f}\n"
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
        ax.set_title(f"Visualisasi Decision Tree C4.5 (Top-{len(expanded_nodes)} Percabangan)", fontsize=13, fontweight='bold', pad=15)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
