import numpy as np

try:
    from .knn import KNN
except ImportError:  # file dijalankan di luar package
    from knn import KNN


class KNNKerabat(KNN):
    """KNN kelompok (knn.py) yang diperluas untuk pemesanan "kerabat".

    Jarak antara query q dan data latih x:

        D(q, x) = ( sum_j |q_j - x_j|^p  +  sum_r penalti_r * beda_r(q, x) ) ^ (1/p)

    - Bagian pertama adalah jarak Minkowski biasa pada fitur numerik (sudah di-scale dan diberi bobot).
    - Bagian kedua adalah jarak atribut simbolik 0/1 (materi kNN): beda_r = 1 jika aturan r dilanggar.
      * Aturan simbolik: kode kedua baris berbeda (kode -1 = tidak diketahui, tidak diberi penalti).
      * Aturan toleransi: selisih nilai kontinu lebih besar dari toleransi (NaN tidak diberi penalti).
    - Suara tetangga dibobot 1 / D^bobot_suara (bobot_suara = 0 berarti suara mayoritas biasa).

    Semua pelanggaran aturan disimpan sebagai bit dalam satu bilangan bulat per pasangan, sehingga total
    penalti cukup diambil dari tabel (lookup table) berukuran 2^R. Dengan cara ini banyak konfigurasi
    penalti dapat dievaluasi dari satu kali perhitungan jarak (dipakai saat tuning).

    Kolom aturan bisa diberikan terpisah (argumen simbol dan kontinu), atau digabung di ujung kanan X
    dengan mengisi n_simbol dan n_kontinu: X = [fitur numerik | kode simbol | nilai kontinu]. Bentuk gabungan
    membuat model bisa dipakai seperti model lain (fit(X, y), predict(X)).

    penalti_relatif: penalti dalam satuan s, yaitu median jarak dasar (pangkat p) ke tetangga ke-10 pada
    sampel data latih. Jika diisi, penalti absolut dihitung ulang setiap kali fit (penalti = penalti_relatif * s).
    """

    _EPS = 1e-9

    def __init__(self, k=25, p=2, bobot_suara=1.0, threshold=0.5, penalti=None, toleransi=None,
                 penalti_relatif=None, n_simbol=0, n_kontinu=0, ukuran_batch=1024, random_state=42):
        super().__init__(k=k, p=p)
        if p < 1:
            raise ValueError('p harus >= 1.')
        self.bobot_suara = float(bobot_suara)
        self.threshold = float(threshold)
        self.penalti = None if penalti is None else np.asarray(penalti, dtype=float)
        self.toleransi = None if toleransi is None else np.asarray(toleransi, dtype=float)
        self.penalti_relatif = None if penalti_relatif is None else np.asarray(penalti_relatif, dtype=float)
        self.n_simbol = int(n_simbol)
        self.n_kontinu = int(n_kontinu)
        self.ukuran_batch = int(ukuran_batch)
        self.random_state = int(random_state)

    def _pisah_gabungan(self, X, simbol, kontinu):
        # Pisahkan X gabungan [numerik | simbol | kontinu] jika kolom aturan tidak diberikan terpisah
        X = np.asarray(X, dtype=float)
        n_aturan = self.n_simbol + self.n_kontinu
        if n_aturan == 0 or simbol is not None or kontinu is not None:
            return X, simbol, kontinu
        d = X.shape[1] - n_aturan
        return X[:, :d], X[:, d:d + self.n_simbol], X[:, d + self.n_simbol:]

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, simbol=None, kontinu=None):
        X, simbol, kontinu = self._pisah_gabungan(X, simbol, kontinu)
        if self.penalti_relatif is not None:
            self.penalti = None
        super().fit(X, y)
        n = len(self.X_train)
        self.y_train = np.asarray(y, dtype=int)
        self.simbol_train = (np.empty((n, 0), dtype=np.int64) if simbol is None
                             else np.asarray(simbol, dtype=np.int64).reshape(n, -1))
        self.kontinu_train = (np.empty((n, 0), dtype=float) if kontinu is None
                              else np.asarray(kontinu, dtype=float).reshape(n, -1))
        n_aturan = self.simbol_train.shape[1] + self.kontinu_train.shape[1]
        if n_aturan > 16:
            raise ValueError('Maksimal 16 aturan beda.')
        if self.penalti is None:
            self.penalti = np.zeros(n_aturan)
        if self.toleransi is None:
            self.toleransi = np.zeros(self.kontinu_train.shape[1])
        self._cek_konfigurasi(self.penalti)
        self._simbol_kolom = np.ascontiguousarray(self.simbol_train.T)
        self._kontinu_kolom = np.ascontiguousarray(self.kontinu_train.T)
        if len(self.toleransi) != self.kontinu_train.shape[1]:
            raise ValueError('Panjang toleransi harus sama dengan jumlah kolom kontinu.')
        self._norm_train = np.einsum('ij,ij->i', self.X_train, self.X_train)
        self._kolom_train = np.ascontiguousarray(self.X_train.T)  # baris = satu fitur, akses berurutan
        if self.penalti_relatif is not None:
            self._cek_konfigurasi(self.penalti_relatif)
            self.penalti = self.penalti_relatif * self.skala_jarak()
        return self

    def skala_jarak(self, n_sampel=500, ke=10):
        """Median jarak dasar (pangkat p) ke tetangga ke-`ke` untuk sampel data latih (tetangga pertama = diri sendiri)."""
        rng = np.random.default_rng(self.random_state)
        pilih = rng.choice(len(self.X_train), size=min(n_sampel, len(self.X_train)), replace=False)
        nol = np.zeros(self.simbol_train.shape[1] + self.kontinu_train.shape[1])
        jarak, _ = self.kneighbors(self.X_train[pilih], self.simbol_train[pilih], self.kontinu_train[pilih],
                                   k=ke + 1, daftar_penalti=[nol])[0]
        return float(np.median(jarak[:, ke] ** self.p))

    def _cek_konfigurasi(self, penalti):
        n_aturan = self.simbol_train.shape[1] + self.kontinu_train.shape[1]
        if len(penalti) != n_aturan:
            raise ValueError(f'Panjang penalti harus {n_aturan} (jumlah kolom simbol + kontinu).')

    # ------------------------------------------------------------- jarak
    def _jarak_dasar(self, Xq):
        """Jumlah |q_j - x_j|^p untuk satu batch query, ukuran (batch, n_train)."""
        if self.p == 2:  # perkalian matriks: |q|^2 + |x|^2 - 2 q.x
            d = self._norm_train[None, :] - 2.0 * (Xq @ self.X_train.T)
            d += np.einsum('ij,ij->i', Xq, Xq)[:, None]
            np.maximum(d, 0.0, out=d)
            d[d < 1e-10] = 0.0
            return d
        d = np.zeros((len(Xq), len(self.X_train)))
        beda = np.empty_like(d)
        for j in range(self.X_train.shape[1]):  # akumulasi per fitur agar hemat memori
            np.subtract(Xq[:, j, None], self._kolom_train[j][None, :], out=beda)
            np.abs(beda, out=beda)
            if self.p != 1:
                np.power(beda, self.p, out=beda)
            d += beda
        return d

    def _bit_beda(self, Sq, Tq):
        """Bit pelanggaran aturan untuk setiap pasangan (batch, n_train)."""
        bit = np.zeros((len(Sq), len(self.X_train)), dtype=np.uint16)
        r = 0
        for j in range(self.simbol_train.shape[1]):
            a, b = Sq[:, j, None], self._simbol_kolom[j][None, :]
            bit |= ((a != b) & (a >= 0) & (b >= 0)).astype(np.uint16) << r
            r += 1
        for j in range(self.kontinu_train.shape[1]):
            selisih = np.abs(Tq[:, j, None] - self._kontinu_kolom[j][None, :])
            bit |= (selisih > self.toleransi[j]).astype(np.uint16) << r  # NaN > t bernilai False
            r += 1
        return bit

    def tabel_penalti(self, penalti):
        """Total penalti untuk setiap kombinasi bit (lookup table berukuran 2^R)."""
        penalti = np.asarray(penalti, dtype=float)
        kombinasi = np.arange(2 ** len(penalti))[:, None]
        return (((kombinasi >> np.arange(len(penalti))) & 1) * penalti).sum(axis=1)

    def _siapkan_query(self, X, simbol, kontinu):
        X, simbol, kontinu = self._pisah_gabungan(X, simbol, kontinu)
        n = len(X)
        S = (np.empty((n, 0), dtype=np.int64) if simbol is None
             else np.asarray(simbol, dtype=np.int64).reshape(n, -1))
        T = (np.empty((n, 0), dtype=float) if kontinu is None
             else np.asarray(kontinu, dtype=float).reshape(n, -1))
        if S.shape[1] != self.simbol_train.shape[1] or T.shape[1] != self.kontinu_train.shape[1]:
            raise ValueError('Jumlah kolom simbol/kontinu query berbeda dengan data latih.')
        return X, S, T

    def kneighbors(self, X, simbol=None, kontinu=None, k=None, daftar_penalti=None):
        """k tetangga terdekat (jarak terurut naik).

        Jika daftar_penalti diisi (list konfigurasi penalti), jarak dasar dan bit pelanggaran dihitung sekali
        per batch, lalu dipakai ulang untuk setiap konfigurasi. Hasilnya list berisi (jarak, indeks).
        """
        if self.X_train is None:
            raise RuntimeError('Model belum di-fit.')
        k = self.k if k is None else int(k)
        k = min(k, len(self.X_train))
        X, S, T = self._siapkan_query(X, simbol, kontinu)
        tunggal = daftar_penalti is None
        daftar = [self.penalti] if tunggal else [np.asarray(c, dtype=float) for c in daftar_penalti]
        for c in daftar:
            self._cek_konfigurasi(c)
        tabel = [self.tabel_penalti(c) for c in daftar]
        ada_aturan = len(daftar[0]) > 0
        hasil = [(np.empty((len(X), k)), np.empty((len(X), k), dtype=np.int64)) for _ in daftar]

        for mulai in range(0, len(X), self.ukuran_batch):
            sl = slice(mulai, mulai + self.ukuran_batch)
            dasar = self._jarak_dasar(X[sl])
            bit = self._bit_beda(S[sl], T[sl]) if ada_aturan else None
            for (jarak_out, idx_out), lut in zip(hasil, tabel):
                total = dasar + lut[bit] if ada_aturan else dasar
                idx = np.argpartition(total, k - 1, axis=1)[:, :k]
                jarak = np.take_along_axis(total, idx, axis=1)
                urut = np.lexsort((idx, jarak), axis=1)      # jarak naik, seri -> indeks kecil dulu
                idx_out[sl] = np.take_along_axis(idx, urut, axis=1)
                jarak_out[sl] = np.take_along_axis(jarak, urut, axis=1) ** (1.0 / self.p)
        return hasil[0] if tunggal else hasil

    # ---------------------------------------------------------- prediksi
    def peluang_dari_tetangga(self, jarak, idx, k=None, bobot_suara=None):
        """Peluang kelas batal dari k tetangga pertama (berbobot 1 / jarak^bobot_suara)."""
        k = self.k if k is None else int(k)
        q = self.bobot_suara if bobot_suara is None else float(bobot_suara)
        jarak, idx = jarak[:, :k], idx[:, :k]
        bobot = np.ones_like(jarak) if q == 0 else 1.0 / (jarak + self._EPS) ** q
        return (bobot * self.y_train[idx]).sum(axis=1) / bobot.sum(axis=1)

    def predict_proba(self, X, simbol=None, kontinu=None):
        jarak, idx = self.kneighbors(X, simbol, kontinu)
        p1 = self.peluang_dari_tetangga(jarak, idx)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, simbol=None, kontinu=None, threshold=None):
        t = self.threshold if threshold is None else float(threshold)
        return (self.predict_proba(X, simbol, kontinu)[:, 1] >= t).astype(int)

    # ------------------------------------------------------ simpan / muat
    def save(self, path):
        relatif = np.full(len(self.penalti), np.nan) if self.penalti_relatif is None else self.penalti_relatif
        np.savez(path, k=self.k, p=self.p, bobot_suara=self.bobot_suara, threshold=self.threshold,
                 penalti=self.penalti, toleransi=self.toleransi, ukuran_batch=self.ukuran_batch,
                 penalti_relatif=relatif, n_simbol=self.n_simbol, n_kontinu=self.n_kontinu,
                 random_state=self.random_state,
                 X_train=self.X_train, y_train=self.y_train,
                 simbol_train=self.simbol_train, kontinu_train=self.kontinu_train)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(k=int(data['k']), p=data['p'].item(), bobot_suara=float(data['bobot_suara']),
                        threshold=float(data['threshold']), penalti=data['penalti'],
                        toleransi=data['toleransi'], ukuran_batch=int(data['ukuran_batch']),
                        n_simbol=int(data['n_simbol']), n_kontinu=int(data['n_kontinu']),
                        random_state=int(data['random_state']))
            # Penalti absolut yang tersimpan dipakai langsung (tidak dihitung ulang)
            model.fit(data['X_train'], data['y_train'], data['simbol_train'], data['kontinu_train'])
            relatif = data['penalti_relatif']
            model.penalti_relatif = None if np.isnan(relatif).all() else relatif
        return model
