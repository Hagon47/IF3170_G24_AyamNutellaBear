import numpy as np

try:
    from .knn import KNN
except ImportError:
    from knn import KNN


class KNNTuning(KNN):
    # X gabungan berurutan: fitur numerik, kode simbol, nilai kontinu.

    _EPS = 1e-9

    def __init__(
        self, k=25, p=2, bobot_suara=1.0, threshold=0.5,
        penalti=None, toleransi=None, penalti_relatif=None,
        n_simbol=0, n_kontinu=0, ukuran_batch=1024, random_state=42,
    ):
        super().__init__(k=k, p=p)
        if p < 1:
            raise ValueError("p harus >= 1.")

        self.bobot_suara = float(bobot_suara)
        self.threshold = float(threshold)
        self.penalti = None if penalti is None else np.asarray(penalti, dtype=float)
        self.toleransi = None if toleransi is None else np.asarray(toleransi, dtype=float)
        self.penalti_relatif = None if penalti_relatif is None else np.asarray(penalti_relatif, dtype=float)
        self.n_simbol = int(n_simbol)
        self.n_kontinu = int(n_kontinu)
        self.ukuran_batch = int(ukuran_batch)
        self.random_state = int(random_state)

    def fit(self, X, y, simbol=None, kontinu=None):
        X, self.simbol_train, self.kontinu_train = self._pisah_fitur(X, simbol, kontinu)
        super().fit(X, np.asarray(y, dtype=int))

        n_aturan = self.simbol_train.shape[1] + self.kontinu_train.shape[1]
        
        if n_aturan > 16:
            raise ValueError("Maksimal 16 aturan beda.")
        
        if self.penalti is None or self.penalti_relatif is not None:
            self.penalti = np.zeros(n_aturan)
        
        if self.toleransi is None:
            self.toleransi = np.zeros(self.kontinu_train.shape[1])

        self._cek_konfigurasi(self.penalti)
        self._simbol_kolom = np.ascontiguousarray(self.simbol_train.T)
        self._kontinu_kolom = np.ascontiguousarray(self.kontinu_train.T)
        
        if len(self.toleransi) != self.kontinu_train.shape[1]:
            raise ValueError("Panjang toleransi harus sama dengan jumlah kolom kontinu.")

        self._norm_train = np.einsum('ij,ij->i', self.X_train, self.X_train)
        self._kolom_train = np.ascontiguousarray(self.X_train.T)
        if self.penalti_relatif is not None:
            self._cek_konfigurasi(self.penalti_relatif)
            self.penalti = self.penalti_relatif * self.skala_jarak()
        return self

    def _pisah_fitur(self, X, simbol, kontinu):
        X = np.asarray(X, dtype=float)
        n_aturan = self.n_simbol + self.n_kontinu
        
        if n_aturan and simbol is None and kontinu is None:
            batas_numerik = X.shape[1] - n_aturan
            batas_simbol = batas_numerik + self.n_simbol
            X, simbol, kontinu = np.split(X, [batas_numerik, batas_simbol], axis=1)

        n = len(X)
        
        simbol = (
            np.empty((n, 0), dtype=np.int64) if simbol is None
            else np.asarray(simbol, dtype=np.int64).reshape(n, -1)
        )
        
        kontinu = (
            np.empty((n, 0), dtype=float) if kontinu is None
            else np.asarray(kontinu, dtype=float).reshape(n, -1)
        )
        
        return X, simbol, kontinu

    def _cek_konfigurasi(self, penalti):
        n_aturan = self.simbol_train.shape[1] + self.kontinu_train.shape[1]
        
        if len(penalti) != n_aturan:
            raise ValueError(f"Panjang penalti harus {n_aturan} (jumlah kolom simbol + kontinu).")

    def _jarak_dasar(self, Xq):
        # Jumlah |q - x|^p sebelum ditambah penalti dan diakarkan.
        
        if self.p == 2:
            distances = self._norm_train[None, :] - 2.0 * (Xq @ self.X_train.T)
            distances += np.einsum('ij,ij->i', Xq, Xq)[:, None]
            distances[distances < 1e-10] = 0.0
            return distances

        distances = np.zeros((len(Xq), len(self.X_train)))
        differences = np.empty_like(distances)
        
        for j in range(self.X_train.shape[1]):
            np.subtract(Xq[:, j, None], self._kolom_train[j][None, :], out=differences)
            np.abs(differences, out=differences)
            if self.p != 1:
                np.power(differences, self.p, out=differences)
            distances += differences
        
        return distances

    def _bit_beda(self, Sq, Tq):
        bit = np.zeros((len(Sq), len(self.X_train)), dtype=np.uint16)
        posisi = 0
        
        for j in range(self.simbol_train.shape[1]):
            a, b = Sq[:, j, None], self._simbol_kolom[j][None, :]
            bit |= ((a != b) & (a >= 0) & (b >= 0)).astype(np.uint16) << posisi
            posisi += 1
        
        for j in range(self.kontinu_train.shape[1]):
            selisih = np.abs(Tq[:, j, None] - self._kontinu_kolom[j][None, :])
            bit |= (selisih > self.toleransi[j]).astype(np.uint16) << posisi
            posisi += 1
        
        return bit

    def tabel_penalti(self, penalti):
        # Total penalti untuk setiap kombinasi bit (lookup table berukuran 2^R).
        penalti = np.asarray(penalti, dtype=float)
        kombinasi = np.arange(2 ** len(penalti))[:, None]
        aturan_aktif = (kombinasi >> np.arange(len(penalti))) & 1
        return (aturan_aktif * penalti).sum(axis=1)

    def kneighbors(self, X, simbol=None, kontinu=None, k=None, daftar_penalti=None):
        # Kembalikan (jarak, indeks), atau daftar pasangan jika penalti divariasikan."""
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model belum di-fit.")

        k = self.k if k is None else int(k)
        k = min(k, len(self.X_train))
        X, simbol, kontinu = self._pisah_fitur(X, simbol, kontinu)
        if (
            simbol.shape[1] != self.simbol_train.shape[1]
            or kontinu.shape[1] != self.kontinu_train.shape[1]
        ):
            raise ValueError("Jumlah kolom simbol/kontinu query berbeda dengan data latih.")

        tunggal = daftar_penalti is None
        konfigurasi = (
            [self.penalti] if tunggal
            else [np.asarray(penalti, dtype=float) for penalti in daftar_penalti]
        )
        for penalti in konfigurasi:
            self._cek_konfigurasi(penalti)
        tabel = [self.tabel_penalti(penalti) if np.any(penalti) else None for penalti in konfigurasi]
        ada_penalti = any(penalti is not None for penalti in tabel)
        hasil = [
            (np.empty((len(X), k)), np.empty((len(X), k), dtype=np.int64))
            for _ in konfigurasi
        ]

        for mulai in range(0, len(X), self.ukuran_batch):
            batch = slice(mulai, mulai + self.ukuran_batch)
            dasar = self._jarak_dasar(X[batch])
            bit = self._bit_beda(simbol[batch], kontinu[batch]) if ada_penalti else None
            for (jarak_out, indeks_out), tabel_konfigurasi in zip(hasil, tabel):
                total = dasar if tabel_konfigurasi is None else dasar + tabel_konfigurasi[bit]
                top_neighbor = np.argpartition(total, k - 1, axis=1)[:, :k]
                jarak = np.take_along_axis(total, top_neighbor, axis=1)
                urut = np.lexsort((top_neighbor, jarak), axis=1)
                indeks_out[batch] = np.take_along_axis(top_neighbor, urut, axis=1)
                jarak_out[batch] = np.take_along_axis(jarak, urut, axis=1) ** (1.0 / self.p)
        return hasil[0] if tunggal else hasil

    def skala_jarak(self, n_sampel=500, ke=10):
        # Median jarak berpangkat p ke tetangga ke-`ke`, atau yang terakhir tersedia.
        rng = np.random.default_rng(self.random_state)
        n_train = len(self.X_train)
        pilih = rng.choice(n_train, size=min(n_sampel, n_train), replace=False)
        nol = np.zeros(self.simbol_train.shape[1] + self.kontinu_train.shape[1])
        
        jarak, _ = self.kneighbors(
            self.X_train[pilih], self.simbol_train[pilih], self.kontinu_train[pilih],
            k=ke + 1, daftar_penalti=[nol],
        )[0]
        
        return float(np.median(jarak[:, -1] ** self.p))

    def peluang_dari_tetangga(self, jarak, idx, k=None, bobot_suara=None):
        k = self.k if k is None else int(k)
        q = self.bobot_suara if bobot_suara is None else float(bobot_suara)
        jarak = jarak[:, :k]
        bobot = np.ones_like(jarak) if q == 0 else 1.0 / (jarak + self._EPS) ** q
        labels = self.y_train[idx[:, :k]]
        
        return (bobot * labels).sum(axis=1) / bobot.sum(axis=1)

    def predict_proba(self, X, simbol=None, kontinu=None):
        jarak, idx = self.kneighbors(X, simbol, kontinu)
        peluang_batal = self.peluang_dari_tetangga(jarak, idx)
        
        return np.column_stack([1.0 - peluang_batal, peluang_batal])

    def predict(self, X, simbol=None, kontinu=None, threshold=None):
        threshold = self.threshold if threshold is None else float(threshold)
        peluang_batal = self.predict_proba(X, simbol, kontinu)[:, 1]
        
        return (peluang_batal >= threshold).astype(int)

    def save(self, path):
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model belum di-fit.")
        
        relatif = np.full(len(self.penalti), np.nan) if self.penalti_relatif is None else self.penalti_relatif
        
        np.savez(
            path, k=self.k, p=self.p, bobot_suara=self.bobot_suara, threshold=self.threshold,
            penalti=self.penalti, toleransi=self.toleransi, ukuran_batch=self.ukuran_batch,
            penalti_relatif=relatif, n_simbol=self.n_simbol, n_kontinu=self.n_kontinu,
            random_state=self.random_state, X_train=self.X_train, y_train=self.y_train,
            simbol_train=self.simbol_train, kontinu_train=self.kontinu_train,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(
                k=int(data["k"]), p=data["p"].item(), bobot_suara=float(data["bobot_suara"]),
                threshold=float(data["threshold"]), penalti=data["penalti"],
                toleransi=data["toleransi"], ukuran_batch=int(data["ukuran_batch"]),
                n_simbol=int(data["n_simbol"]), n_kontinu=int(data["n_kontinu"]),
                random_state=int(data["random_state"]),
            )
            
            # Pasang penalti relatif setelah fit agar penalti absolut tersimpan dipakai langsung.
            model.fit(data["X_train"], data["y_train"], data["simbol_train"], data["kontinu_train"])
            relatif = data["penalti_relatif"]
            model.penalti_relatif = None if np.isnan(relatif).all() else relatif
      
        return model
