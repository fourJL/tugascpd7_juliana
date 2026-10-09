# Verifikasi Nomor Ijazah dan Deteksi Tanda Tangan

Program Python ini memproses kumpulan gambar ijazah untuk:

- membaca kandidat nomor ijazah menggunakan Tesseract OCR;
- membandingkan tiga metode peningkatan citra: contrast stretching, CLAHE,
  dan gamma correction;
- menghitung Character Error Rate (CER) terhadap nomor ijazah acuan;
- mendeteksi indikasi tanda tangan menggunakan thresholding, morphology, dan
  connected components; serta
- menampilkan ringkasan hasil untuk setiap gambar di terminal.

Program memproses gambar satu per satu. Hasilnya ditampilkan di terminal dan
tidak disimpan sebagai berkas.

## Struktur proyek

```text
citra7/
├── main.py
└── gambar_ijazah/
    ├── gambar_1.jpg
    ├── gambar_2.jpg
    └── ...
```

Folder `gambar_ijazah` harus berada di direktori kerja saat program dijalankan.
Program membaca berkas berformat `.jpg`, `.jpeg`, dan `.png` dari folder itu.
Subfolder tidak dipindai.

## Persyaratan

- Windows
- Python 3
- Tesseract OCR untuk Windows
- Paket Python: `opencv-python`, `numpy`, dan `pytesseract`

## Instalasi

### 1. Pasang Tesseract OCR

Pasang Tesseract OCR dan pastikan file `tesseract.exe` tersedia di salah satu
lokasi yang diperiksa oleh program:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
C:\Program Files (x86)\Tesseract-OCR\tesseract.exe
```

Jika Tesseract dipasang di lokasi lain, ubah daftar `TESSERACT_PATHS` di
`main.py` agar mencantumkan lokasi `tesseract.exe` yang benar. Program berhenti
dengan pesan kesalahan apabila executable tidak ditemukan.

### 2. Siapkan lingkungan Python dan paket

Buka PowerShell di direktori proyek, lalu jalankan:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install opencv-python numpy pytesseract
```

Jika PowerShell menolak aktivasi skrip, Anda dapat menjalankan Python dari
lingkungan virtual secara langsung:

```powershell
.\.venv\Scripts\python.exe -m pip install opencv-python numpy pytesseract
```

## Menjalankan program

Dari direktori proyek (direktori yang berisi `main.py` dan
`gambar_ijazah`), jalankan:

```powershell
python main.py
```

Pastikan folder `gambar_ijazah` berisi setidaknya satu gambar yang didukung.
Jika folder tidak berisi gambar yang didukung, program menampilkan pesan bahwa
tidak ada gambar untuk diproses.

## Konfigurasi

Di bagian konfigurasi `main.py`:

- `FOLDER` menentukan folder gambar yang akan dibaca. Nilai bawaan:
  `gambar_ijazah`.
- `GROUND_TRUTH` adalah nomor ijazah acuan. Ganti nilainya sesuai dataset
  sebelum menjalankan program.
- `TESSERACT_PATHS` berisi lokasi-lokasi `tesseract.exe` yang akan diperiksa.

`GROUND_TRUTH` berpengaruh pada dua hal: pemilihan kandidat angka oleh
panjangnya yang paling mendekati nomor acuan, serta penghitungan CER. Program
juga memilih enhancement dengan CER terkecil. Karena itu, hasil pilihan
enhancement menggunakan nomor acuan dan bukan evaluasi independen tanpa
label. Gunakan nomor acuan yang benar untuk dataset yang sedang dianalisis.

## Alur pemrosesan

Untuk setiap gambar, program:

1. Membaca gambar dan mengubahnya menjadi grayscale.
2. Membuat tiga versi enhancement: contrast stretching, CLAHE, dan gamma
   correction dengan gamma `1.5`.
3. Menjalankan OCR pada setiap versi dengan konfigurasi Tesseract `--psm 6`.
4. Mengambil rangkaian angka yang memiliki setidaknya 10 digit. Jika ada
   beberapa kandidat, dipilih kandidat dengan panjang yang paling dekat dengan
   panjang `GROUND_TRUTH`.
5. Menghitung CER dari jarak Levenshtein kandidat terhadap `GROUND_TRUTH`, lalu
   memilih versi enhancement dengan CER terkecil.
6. Menandai area teks yang terdeteksi OCR, lalu menjalankan deteksi tanda
   tangan dengan Otsu dan adaptive threshold. Hasil dipilih berdasarkan skor
   kandidat, dengan persentase area sebagai pemecah nilai seri.
7. Menampilkan nomor yang diprediksi, CER, enhancement terpilih, status tanda
   tangan, metode deteksi, dan persentase area tanda tangan di terminal.

Deteksi tanda tangan merupakan heuristik berbasis ukuran, rasio aspek, luas,
dan kepadatan komponen citra setelah pemrosesan; hasil `PRESENT` atau `ABSENT`
bukan verifikasi tanda tangan secara biometrik.

## Memahami keluaran

- **Nomor Ijazah**: kandidat angka yang diekstrak dari OCR.
- **CER (%)**: jarak edit Levenshtein antara kandidat dan `GROUND_TRUTH`,
  dibagi jumlah karakter pada `GROUND_TRUTH`, lalu ditampilkan sebagai
  persentase. Nilai lebih rendah berarti lebih dekat dengan acuan. Jika OCR
  tidak menghasilkan kandidat, CER adalah `100%`.
- **Enhancement Terbaik**: metode peningkatan citra yang menghasilkan CER
  paling kecil terhadap acuan.
- **Tanda Tangan**: `PRESENT` jika deteksi menemukan komponen yang memenuhi
  kriteria program; selain itu `ABSENT`.
- **Metode**: metode thresholding yang dipilih, yaitu `Otsu` atau
  `Adaptive Threshold`.
- **Persentase Area**: luas komponen tanda tangan terpilih dibanding luas
  seluruh gambar, dalam persen. Nilainya `0%` jika tidak ditemukan kandidat.

## Pemecahan masalah

- **`Tesseract tidak ditemukan`**: pasang Tesseract OCR atau tambahkan lokasi
  `tesseract.exe` ke `TESSERACT_PATHS`.
- **`Tidak ada gambar di folder`**: periksa bahwa `gambar_ijazah` berada di
  direktori kerja saat ini dan berisi berkas `.jpg`, `.jpeg`, atau `.png`.
- **`Gagal membaca` gambar**: pastikan berkas gambar dapat dibuka dan tidak
  rusak.
- **Hasil OCR atau deteksi tidak sesuai**: verifikasi `GROUND_TRUTH`, kualitas
  gambar, dan lokasi teks/tanda tangan. Algoritma saat ini menggunakan ambang
  dan kriteria heuristik tetap di `main.py`, sehingga hasil dapat bervariasi
  untuk gambar dengan karakteristik berbeda.
