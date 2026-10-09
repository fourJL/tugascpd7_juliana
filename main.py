import cv2
import os
import re
import csv
import numpy as np
import pytesseract
from pytesseract import Output


# =========================================================
# KONFIGURASI
# =========================================================

FOLDER = "gambar_ijazah"

# Semua output akan disimpan di folder ini
OUTPUT_FOLDER = "hasil"

# Nomor ijazah sebenarnya.
# GANTI sesuai nomor ijazah pada dataset kamu.
GROUND_TRUTH = "571012022000056"


# =========================================================
# KONFIGURASI TESSERACT
# =========================================================

TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
]

tesseract_found = False

for path in TESSERACT_PATHS:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        tesseract_found = True
        break

if not tesseract_found:
    print("ERROR: Tesseract tidak ditemukan!")
    print("Pastikan Tesseract sudah terinstall.")
    exit()

# Siapkan folder output secara otomatis
FOLDER_GRAYSCALE = os.path.join(OUTPUT_FOLDER, "grayscale")
FOLDER_ENHANCEMENT = os.path.join(OUTPUT_FOLDER, "enhancement")
FOLDER_CONTRAST = os.path.join(FOLDER_ENHANCEMENT, "contrast_stretching")
FOLDER_CLAHE = os.path.join(FOLDER_ENHANCEMENT, "clahe")
FOLDER_GAMMA = os.path.join(FOLDER_ENHANCEMENT, "gamma_correction")
FOLDER_TERBAIK = os.path.join(OUTPUT_FOLDER, "enhancement_terbaik")
FOLDER_MASK = os.path.join(OUTPUT_FOLDER, "text_mask")
FOLDER_DETEKSI = os.path.join(OUTPUT_FOLDER, "deteksi_tanda_tangan")

for folder_output in [
    OUTPUT_FOLDER, FOLDER_GRAYSCALE, FOLDER_CONTRAST, FOLDER_CLAHE,
    FOLDER_GAMMA, FOLDER_TERBAIK, FOLDER_MASK, FOLDER_DETEKSI
]:
    os.makedirs(folder_output, exist_ok=True)


# =========================================================
# 1. GRAYSCALE
# =========================================================

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# =========================================================
# 2. IMAGE ENHANCEMENT
# =========================================================

def contrast_stretch(img):
    minimum = np.min(img)
    maximum = np.max(img)

    if maximum == minimum:
        return img

    result = ((img - minimum) * 255 / (maximum - minimum))

    return result.astype(np.uint8)


def clahe(img):
    clahe_object = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe_object.apply(img)


def gamma_correction(img, gamma=1.5):

    table = np.array([
        ((i / 255.0) ** (1 / gamma)) * 255
        for i in range(256)
    ]).astype(np.uint8)

    return cv2.LUT(img, table)


# =========================================================
# 3. EKSTRAK NOMOR IJAZAH
# =========================================================

def ekstrak_nomor(text):

    # Hilangkan spasi dan baris baru
    text = text.replace(" ", "")
    text = text.replace("\n", "")

    # Cari angka dengan panjang minimal 10 digit
    kandidat = re.findall(r"\d{10,}", text)

    if not kandidat:
        return ""

    # Pilih angka yang panjangnya paling dekat
    # dengan nomor ijazah sebenarnya
    return min(
        kandidat,
        key=lambda x: abs(
            len(x) - len(GROUND_TRUTH)
        )
    )


# =========================================================
# 4. LEVENSHTEIN DISTANCE
# =========================================================

def levenshtein(a, b):

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, 1):

        current = [i]

        for j, char_b in enumerate(b, 1):

            if char_a == char_b:
                cost = 0
            else:
                cost = 1

            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost
                )
            )

        previous = current

    return previous[-1]


# =========================================================
# 5. MENGHITUNG CER
# =========================================================

def hitung_cer(prediksi):

    if not prediksi:
        return 1.0

    distance = levenshtein(
        prediksi,
        GROUND_TRUTH
    )

    return distance / len(GROUND_TRUTH)


# =========================================================
# 6. OCR NOMOR IJAZAH
# =========================================================

def ocr_nomor(image):

    text = pytesseract.image_to_string(
        image,
        config="--psm 6"
    )

    nomor = ekstrak_nomor(text)

    return nomor


# =========================================================
# 7. MEMBUAT TEXT MASK
#    Digunakan untuk menghilangkan teks sebelum
#    proses deteksi tanda tangan
# =========================================================

def buat_text_mask(gray):

    data = pytesseract.image_to_data(
        gray,
        config="--psm 6",
        output_type=Output.DICT
    )

    mask = np.zeros(
        gray.shape[:2],
        dtype=np.uint8
    )

    for i, text in enumerate(data["text"]):

        text = text.strip()

        if not text:
            continue

        try:
            confidence = float(
                data["conf"][i]
            )
        except:
            confidence = 0

        if confidence < 20:
            continue

        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        # Padding supaya area tulisan
        # benar-benar ikut dihapus
        padding = 5

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)

        x2 = min(
            gray.shape[1],
            x + w + padding
        )

        y2 = min(
            gray.shape[0],
            y + h + padding
        )

        mask[y1:y2, x1:x2] = 255

    return mask


# =========================================================
# 8. DETEKSI TANDA TANGAN
#    THRESHOLDING + MORPHOLOGY +
#    CONNECTED COMPONENTS
# =========================================================

def deteksi_metode(
    gray,
    text_mask,
    adaptive=False,
    output_path=None
):

    # -----------------------------------------------------
    # THRESHOLDING
    # -----------------------------------------------------

    if adaptive:

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15
        )

    else:

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV +
            cv2.THRESH_OTSU
        )

    # -----------------------------------------------------
    # HILANGKAN TEKS
    # -----------------------------------------------------

    binary[text_mask == 255] = 0

    # -----------------------------------------------------
    # MORPHOLOGY
    # -----------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    # Opening
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Closing
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Simpan citra biner hasil thresholding dan morphology jika diminta
    if output_path is not None:
        cv2.imwrite(output_path, binary)

    # -----------------------------------------------------
    # CONNECTED COMPONENTS
    # -----------------------------------------------------

    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            binary,
            8
        )

    height, width = gray.shape

    total_area = height * width

    candidates = []

    # -----------------------------------------------------
    # MENCARI OBJEK YANG MENYERUPAI TANDA TANGAN
    # -----------------------------------------------------

    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        x = stats[
            i,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            i,
            cv2.CC_STAT_TOP
        ]

        object_width = stats[
            i,
            cv2.CC_STAT_WIDTH
        ]

        object_height = stats[
            i,
            cv2.CC_STAT_HEIGHT
        ]

        # Abaikan noise kecil
        if area < total_area * 0.0003:
            continue

        # Objek harus cukup lebar
        if object_width < width * 0.08:
            continue

        # -------------------------------------------------
        # KARAKTERISTIK OBJEK
        # -------------------------------------------------

        aspect_ratio = (
            object_width /
            max(object_height, 1)
        )

        density = (
            area /
            max(
                object_width *
                object_height,
                1
            )
        )

        # -------------------------------------------------
        # SCORING
        # -------------------------------------------------

        score = 0

        # Lebar objek
        if object_width >= width * 0.15:
            score += 2

        # Bentuk memanjang
        if aspect_ratio >= 2:
            score += 2

        # Luas objek
        if area >= total_area * 0.001:
            score += 1

        # Tanda tangan biasanya tidak terlalu padat
        if density < 0.50:
            score += 1

        # Jika memenuhi kriteria
        if score >= 4:

            candidates.append(
                (
                    area,
                    object_width,
                    object_height,
                    aspect_ratio,
                    density,
                    score
                )
            )

    # -----------------------------------------------------
    # HASIL DETEKSI
    # -----------------------------------------------------

    if candidates:

        best = max(
            candidates,
            key=lambda x: (
                x[5],
                x[0]
            )
        )

        percentage = (
            best[0] /
            total_area
        ) * 100

        return (
            "PRESENT",
            percentage,
            best[5]
        )

    return (
        "ABSENT",
        0.0,
        0
    )


# =========================================================
# 9. MEMBANDINGKAN OTSU DAN ADAPTIVE THRESHOLD
# =========================================================

def deteksi_tanda_tangan(gray, nama_file):

    # OCR digunakan untuk mendapatkan
    # area teks yang akan dihilangkan
    text_mask = buat_text_mask(gray)
    nama_dasar = os.path.splitext(nama_file)[0]
    cv2.imwrite(os.path.join(FOLDER_MASK, f"{nama_dasar}_text_mask.png"), text_mask)

    # Otsu
    hasil_otsu = deteksi_metode(
        gray,
        text_mask,
        adaptive=False,
        output_path=os.path.join(FOLDER_DETEKSI, f"{nama_dasar}_otsu.png")
    )

    # Adaptive Threshold
    hasil_adaptive = deteksi_metode(
        gray,
        text_mask,
        adaptive=True,
        output_path=os.path.join(FOLDER_DETEKSI, f"{nama_dasar}_adaptive.png")
    )

    # -----------------------------------------------------
    # PILIH HASIL DENGAN SCORE TERBAIK
    # -----------------------------------------------------

    if hasil_adaptive[2] > hasil_otsu[2]:

        return (
            hasil_adaptive[0],
            hasil_adaptive[1],
            "Adaptive Threshold"
        )

    elif hasil_otsu[2] > hasil_adaptive[2]:

        return (
            hasil_otsu[0],
            hasil_otsu[1],
            "Otsu"
        )

    else:

        # Jika score sama,
        # pilih persentase terbesar

        if hasil_adaptive[1] > hasil_otsu[1]:

            return (
                hasil_adaptive[0],
                hasil_adaptive[1],
                "Adaptive Threshold"
            )

        return (
            hasil_otsu[0],
            hasil_otsu[1],
            "Otsu"
        )


# =========================================================
# 10. PROSES SEMUA GAMBAR
# =========================================================

hasil = []

files = sorted([
    file
    for file in os.listdir(FOLDER)
    if file.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png"
        )
    )
])


if len(files) == 0:

    print(
        "Tidak ada gambar di folder:",
        FOLDER
    )

    exit()


# =========================================================
# LOOP SETIAP GAMBAR
# =========================================================

for filename in files:

    path = os.path.join(
        FOLDER,
        filename
    )

    image = cv2.imread(path)

    if image is None:

        print(
            "Gagal membaca:",
            filename
        )

        continue

    # -----------------------------------------------------
    # GRAYSCALE
    # -----------------------------------------------------

    gray = grayscale(image)
    nama_dasar = os.path.splitext(filename)[0]
    cv2.imwrite(os.path.join(FOLDER_GRAYSCALE, f"{nama_dasar}_grayscale.png"), gray)

    # -----------------------------------------------------
    # IMAGE ENHANCEMENT
    # -----------------------------------------------------

    enhancement = {

        "Contrast Stretching":
            contrast_stretch(gray),

        "CLAHE":
            clahe(gray),

        "Gamma Correction":
            gamma_correction(
                gray,
                gamma=1.5
            )
    }

    # Simpan semua citra hasil enhancement
    cv2.imwrite(os.path.join(FOLDER_CONTRAST, f"{nama_dasar}_contrast_stretching.png"), enhancement["Contrast Stretching"])
    cv2.imwrite(os.path.join(FOLDER_CLAHE, f"{nama_dasar}_clahe.png"), enhancement["CLAHE"])
    cv2.imwrite(os.path.join(FOLDER_GAMMA, f"{nama_dasar}_gamma_correction.png"), enhancement["Gamma Correction"])

    hasil_ocr = []

    # -----------------------------------------------------
    # OCR DENGAN 3 ENHANCEMENT
    # -----------------------------------------------------

    for nama_metode, enhanced in enhancement.items():

        nomor = ocr_nomor(
            enhanced
        )

        cer = hitung_cer(
            nomor
        )

        hasil_ocr.append(
            (
                nama_metode,
                nomor,
                cer
            )
        )

    # -----------------------------------------------------
    # PILIH ENHANCEMENT DENGAN CER TERKECIL
    # -----------------------------------------------------

    terbaik = min(
        hasil_ocr,
        key=lambda x: x[2]
    )
    cv2.imwrite(
        os.path.join(FOLDER_TERBAIK, f"{nama_dasar}_enhancement_terbaik.png"),
        enhancement[terbaik[0]]
    )

    # -----------------------------------------------------
    # DETEKSI TANDA TANGAN
    # -----------------------------------------------------

    signature, percentage, metode_signature = \
        deteksi_tanda_tangan(gray, filename)

    # -----------------------------------------------------
    # SIMPAN HASIL
    # -----------------------------------------------------

    hasil.append(
        [
            filename,
            terbaik[1],
            terbaik[2] * 100,
            terbaik[0],
            signature,
            percentage,
            metode_signature
        ]
    )


# =========================================================
# 11. OUTPUT HASIL
# =========================================================

print("\n")
print("=" * 105)

print(
    f"{'Nama Gambar':<20}"
    f"{'Nomor Ijazah':<22}"
    f"{'CER (%)':<12}"
    f"{'Enhancement':<25}"
    f"{'Tanda Tangan':<17}"
    f"{'Metode':<20}"
)

print("=" * 105)


for row in hasil:

    print(
        f"{row[0]:<20}"
        f"{row[1]:<22}"
        f"{row[2]:<12.2f}"
        f"{row[3]:<25}"
        f"{row[4]:<17}"
        f"{row[6]:<20}"
    )


print("=" * 105)


# =========================================================
# 12. KESIMPULAN
# =========================================================

print("\nHASIL VERIFIKASI:")

for row in hasil:

    print(
        f"\nFile : {row[0]}"
    )

    print(
        f"Nomor Ijazah : {row[1]}"
    )

    print(
        f"CER : {row[2]:.2f}%"
    )

    print(
        f"Enhancement Terbaik : {row[3]}"
    )

    print(
        f"Tanda Tangan : {row[4]}"
    )

    print(
        f"Metode Deteksi : {row[6]}"
    )

    print(
        f"Persentase Area : {row[5]:.2f}%"
    )

print("\nProses selesai.")


# =========================================================
# 13. SIMPAN HASIL VERIFIKASI KE CSV
# =========================================================

file_csv = os.path.join(OUTPUT_FOLDER, "hasil_verifikasi.csv")

with open(file_csv, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Nama Gambar",
        "Nomor Ijazah OCR",
        "CER (%)",
        "Enhancement Terbaik",
        "Status Tanda Tangan",
        "Persentase Area (%)",
        "Metode Deteksi Tanda Tangan"
    ])

    for row in hasil:
        writer.writerow([
            row[0],
            row[1],
            f"{row[2]:.2f}",
            row[3],
            row[4],
            f"{row[5]:.2f}",
            row[6]
        ])

print(f"\nFile CSV berhasil disimpan: {file_csv}")
print(f"Semua citra hasil disimpan di folder: {OUTPUT_FOLDER}/")
