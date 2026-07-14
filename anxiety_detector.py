import argparse
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# KONFIGURASI & AMBANG (THRESHOLD)
# Seluruh ambang dinormalisasi terhadap jarak antar-mata
# (Inter-Ocular Distance / IOD) agar bersifat scale-invariant:
# hasil tidak berubah ketika subjek mendekat/menjauh dari kamera.
# ============================================================
class Config:
    # --- Fidgeting (gerakan kepala berlebih) ---
    # Rata-rata perpindahan hidung per-frame (satuan IOD) di atas nilai ini,
    FIDGET_THRESHOLD = 0.07
    HEAD_WINDOW = 10  # jumlah frame untuk merata-ratakan perpindahan kepala

    # --- Face Touching (menyentuh wajah) ---
    # Jarak minimum ujung jari ke landmark wajah (satuan IOD).
    FACE_TOUCH_THRESHOLD = 0.65

    # --- Penghalusan status (temporal smoothing) agar teks tidak berkedip ---
    SMOOTH_WINDOW = 15   # jumlah frame terakhir yang dievaluasi
    ANXIOUS_RATIO = 0.40  # proporsi frame "cemas" minimum agar status = CEMAS

    # --- Indeks landmark wajah (MediaPipe Face Mesh, 468 titik) ---
    NOSE_TIP = 1
    RIGHT_EYE_OUTER = 33
    LEFT_EYE_OUTER = 263

    # --- Indeks ujung jari (MediaPipe Hands, 21 titik per tangan) ---
    FINGERTIPS = (4, 8, 12, 16, 20)  # jempol, telunjuk, tengah, manis, kelingking


# ============================================================
# FUNGSI BANTU
# ============================================================
def euclidean(p1, p2):
    """Jarak Euclidean antara dua titik 2D (piksel)."""
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def to_pixel(landmark, w, h):
    """Konversi landmark ternormalisasi MediaPipe (0..1) ke koordinat piksel."""
    return np.array([landmark.x * w, landmark.y * h], dtype=np.float32)


def preprocess_frame(frame, clahe):
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = clahe.apply(y)  # equalisasi adaptif hanya pada luminans
    merged = cv2.merge((y, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


# ============================================================
# DETEKTOR
# ============================================================
class AnxietyDetector:
    def __init__(self, cfg, draw_landmarks=False, use_preprocess=True):
        self.cfg = cfg
        self.draw_landmarks = draw_landmarks
        self.use_preprocess = use_preprocess

        # Modul MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Satu graf Holistic mencakup deteksi wajah + tangan sekaligus.
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # CLAHE: clipLimit membatasi amplifikasi noise; tileGridSize = ukuran region lokal.
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Buffer temporal
        self.head_disp = deque(maxlen=cfg.HEAD_WINDOW)     # riwayat perpindahan kepala
        self.status_buf = deque(maxlen=cfg.SMOOTH_WINDOW)  # riwayat status cemas/tenang
        self.prev_nose = None                              # posisi hidung frame sebelumnya

    def close(self):
        self.holistic.close()

    # ---- Fitur 1: jarak antar-mata sebagai skala referensi --------------
    def _inter_ocular(self, face, w, h):
        r = to_pixel(face[self.cfg.RIGHT_EYE_OUTER], w, h)
        l = to_pixel(face[self.cfg.LEFT_EYE_OUTER], w, h)
        return max(euclidean(r, l), 1e-6)  # cegah pembagian nol

    # ---- Fitur 2: deteksi fidgeting -------------------------------------
    def _detect_fidget(self, face, iod, w, h):
        nose = to_pixel(face[self.cfg.NOSE_TIP], w, h)
        if self.prev_nose is not None:
            disp = euclidean(nose, self.prev_nose) / iod  # perpindahan ternormalisasi
            self.head_disp.append(disp)
        self.prev_nose = nose

        if len(self.head_disp) < self.head_disp.maxlen:
            return False, 0.0  # buffer belum penuh
        avg = float(np.mean(self.head_disp))
        return avg > self.cfg.FIDGET_THRESHOLD, avg

    # ---- Fitur 3: deteksi menyentuh wajah -------------------------------
    def _detect_face_touch(self, face, hands_landmarks, iod, w, h):
        if not hands_landmarks:
            return False, None
        face_pts = np.array([[lm.x * w, lm.y * h] for lm in face], dtype=np.float32)
        min_norm = np.inf
        for hand in hands_landmarks:
            for idx in self.cfg.FINGERTIPS:
                tip = to_pixel(hand.landmark[idx], w, h)
                # jarak jari ke landmark wajah terdekat, dinormalisasi terhadap IOD
                d = float(np.min(np.linalg.norm(face_pts - tip, axis=1))) / iod
                min_norm = min(min_norm, d)
        if not np.isfinite(min_norm):
            return False, None
        return min_norm < self.cfg.FACE_TOUCH_THRESHOLD, min_norm

    # ---- Pipeline per-frame ---------------------------------------------
    def process(self, frame):
        h, w = frame.shape[:2]

        proc = preprocess_frame(frame, self.clahe) if self.use_preprocess else frame
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # optimasi: hindari salinan internal
        results = self.holistic.process(rgb)

        fidget, touch = False, False
        metrics = {"head_avg": 0.0, "touch_dist": None}

        if results.face_landmarks:
            face = results.face_landmarks.landmark
            iod = self._inter_ocular(face, w, h)

            fidget, metrics["head_avg"] = self._detect_fidget(face, iod, w, h)

            hands = [hlm for hlm in
                     (results.left_hand_landmarks, results.right_hand_landmarks)
                     if hlm is not None]
            touch, metrics["touch_dist"] = self._detect_face_touch(face, hands, iod, w, h)
        else:
            # Wajah hilang dari frame: reset acuan agar tidak salah menghitung lonjakan
            # perpindahan saat wajah muncul kembali.
            self.prev_nose = None
            self.head_disp.clear()

        # Gabungkan kedua indikator, lalu haluskan secara temporal.
        anxious_now = fidget or touch
        self.status_buf.append(anxious_now)
        ratio = sum(self.status_buf) / len(self.status_buf) if self.status_buf else 0.0
        is_anxious = ratio >= self.cfg.ANXIOUS_RATIO

        if self.draw_landmarks:
            self._draw(frame, results)
        self._overlay(frame, is_anxious, fidget, touch, metrics)

        return frame, is_anxious

    # Visualisasi
    def _draw(self, frame, results):
        d = self.mp_drawing
        if results.face_landmarks:
            d.draw_landmarks(frame, results.face_landmarks,
                              self.mp_face_mesh.FACEMESH_CONTOURS)
        for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
            if hand:
                d.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

    def _overlay(self, frame, is_anxious, fidget, touch, metrics):
        h, w = frame.shape[:2]
        if is_anxious:
            status, color = "INDIKASI GELISAH/CEMAS", (0, 0, 255)  # merah
        else:
            status, color = "TENANG / FOKUS", (0, 180, 0)          # hijau

        # Bilah status atas
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"Status: {status}", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        # Info detail fitur (untuk kalibrasi ambang saat demo)
        flag_f = "ON" if fidget else "off"
        flag_t = "ON" if touch else "off"
        info = f"Fidget[{flag_f}] avg={metrics['head_avg']:.3f} FaceTouch[{flag_t}]"
        if metrics["touch_dist"] is not None:
            info += f" d={metrics['touch_dist']:.2f}"
        cv2.putText(frame, info, (15, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Student Anxiety & Fidgeting Detector (MediaPipe + Rule-Based)")
    parser.add_argument("--camera", type=int, default=0, help="indeks kamera (default 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--draw-landmarks", action="store_true",
                         help="gambar landmark wajah & tangan")
    parser.add_argument("--no-preprocess", action="store_true",
                         help="nonaktifkan pra-pemrosesan CLAHE")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera index {args.camera} tidak dapat dibuka.")

    detector = AnxietyDetector(
        Config,
        draw_landmarks=args.draw_landmarks,
        use_preprocess=not args.no_preprocess,
    )

    prev_t = time.time()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Gagal membaca frame dari kamera.")
                break

            frame = cv2.flip(frame, 1)  # mirror agar terasa seperti cermin
            frame, _ = detector.process(frame)

            # Estimasi FPS dengan exponential moving average
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            cv2.putText(frame, f"FPS: {fps:4.1f}", (frame.shape[1] - 130, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Student Anxiety & Fidgeting Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
