import cv2
from ultralytics import YOLO

# 1. Panggil model YOLOv8 versi paling ringan/cepat
model = YOLO("yolov8n.pt")

# 2. Aktifkan webcam bawaan laptop (angka 0)
kamera = cv2.VideoCapture(0)

print("Program berjalan... Tekan tombol 'q' untuk keluar.")

while True:
    # Membaca gambar dari webcam
    berhasil, frame = kamera.read()
    if not berhasil:
        break

    # 3. Lakukan deteksi objek (Maksimal 3 objek sesuai instruksi tugas)
    hasil_deteksi = model(frame, max_det=3)

    # 4. Gambar kotak & teks secara otomatis menggunakan fungsi bawaan (.plot())
    # Fungsi ini otomatis menggambar Bounding Box, Nama Objek, dan Confidence Score
    frame_hasil = hasil_deteksi[0].plot()

    # 5. Tampilkan hasilnya langsung di layar komputer
    cv2.imshow("Hasil Deteksi Webcam", frame_hasil)

    # Program akan berhenti jika Anda menekan tombol 'q' pada keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. Matikan kamera dan tutup semua jendela setelah selesai
kamera.release()
cv2.destroyAllWindows()