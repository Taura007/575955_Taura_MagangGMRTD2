import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
kamera = cv2.VideoCapture(0)

print("Program berjalan... Tekan tombol 'q' untuk keluar.")

while True:
    berhasil, frame = kamera.read()
    if not berhasil:
        break

    hasil_deteksi = model(frame, max_det=3)
    frame_hasil = hasil_deteksi[0].plot()

    cv2.imshow("Hasil Deteksi Webcam", frame_hasil)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


kamera.release()
cv2.destroyAllWindows()
