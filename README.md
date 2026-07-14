# Real-Time Anxiety & Fidgeting Pattern Recognition

Real-time recognition of anxious/restless behavior patterns in students during study sessions, using webcam video, MediaPipe Holistic landmarks, and geometric rule-based classification — **no trained model, no labeled dataset required.**

> Built for the Pattern Recognition course, Teknik Komputer, Fakultas Ilmu Komputer, Universitas Brawijaya.

![Status: Tenang/Fokus](assets/demo_calm.png) ![Status: Indikasi Gelisah/Cemas](assets/demo_anxious.png)
*(add screenshots/GIF here — see "Adding Media" below)*

## What it does

Points a webcam at a student while they study and classifies their state frame-by-frame into:

- **`TENANG / FOKUS`** (Calm / Focused) — Class 0
- **`INDIKASI GELISAH/CEMAS`** (Indication of Restlessness/Anxiety) — Class 1

based on two behavioral cues tracked in real time:

1. **Fidgeting** — excessive head/nose movement across frames
2. **Face touching** — hand proximity to the face (chin resting, nail biting, etc.)

## How it works

Instead of training a classifier from scratch, this project **repurposes a pre-trained geometric landmarker** (MediaPipe Holistic — Face Mesh + Hand Landmarks) and applies **rule-based thresholding** on spatial distances between landmarks. This keeps the system lightweight, explainable, and free of the need for labeled training data.

### 1. Scale-invariant normalization

All distance thresholds are normalized against **Inter-Ocular Distance (IOD)** — the distance between the outer corners of the eyes — so the system stays accurate regardless of how close/far the subject is from the camera:

```
IOD = ||E_left − E_right||
```

### 2. Feature 1 — Fidgeting detection

Frame-to-frame displacement of the nose tip, normalized by IOD, averaged over a sliding window:

```
d_t = ||N_t − N_(t-1)|| / IOD
D   = (1/W) · Σ d_t          (W = 10-frame window)
```

Flag **fidgeting** if `D > 0.07`.

### 3. Feature 2 — Face-touching detection

Minimum spatial distance from any of the 5 fingertips to the nearest point on the 468-point Face Mesh, normalized by IOD:

```
d_min = min_k,m ||F_k − M_m|| / IOD
```

Flag **face touch** if `d_min < 0.65`.

### 4. Fusion + temporal smoothing

Final decision = `fidgeting OR face_touch`, smoothed over a 15-frame window (`ANXIOUS_RATIO = 0.40`) to prevent label flickering.

| Parameter | Default | Meaning |
|---|---|---|
| `FIDGET_THRESHOLD` | 0.07 | Head-displacement threshold (IOD units) |
| `HEAD_WINDOW` | 10 | Frames averaged for head displacement |
| `FACE_TOUCH_THRESHOLD` | 0.65 | Finger-to-face distance threshold (IOD units) |
| `SMOOTH_WINDOW` / `ANXIOUS_RATIO` | 15 / 0.40 | Temporal smoothing to avoid flicker |

### 5. Preprocessing for lighting robustness

Variable room lighting causes low contrast and jittery landmark detection. Before each frame reaches MediaPipe:

1. **Gaussian Blur (3×3)** — low-pass filter to suppress sensor/exposure noise
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) — applied only to the luminance (Y) channel in YCrCb space, stabilizing local contrast per-tile without amplifying noise; chroma channels (Cr, Cb) are left untouched

Reference: Zuiderveld (1994, *Graphics Gems IV*), Pizer et al. (1987), Gonzalez & Woods (*Digital Image Processing*). CLAHE is histogram/LUT-based with linear complexity, so it adds negligible overhead to the real-time pipeline.

## Why this runs on a local PC, not an embedded MCU

The pipeline was deliberately kept on **edge computing (local PC)** rather than a microcontroller like the ESP32, because MediaPipe's cascaded CNNs (Face Mesh, 468 points + Hand Landmarks) are too resource-heavy for constrained hardware:

| Aspect | ESP32 | Pipeline Requirement |
|---|---|---|
| RAM | ~520 KB internal SRAM | CNN activation buffers + framebuffer reach MB-scale → OOM risk |
| Flash | ~4 MB | MediaPipe model weights reach tens of MB — won't fit |
| Accelerator | Scalar FPU, no NPU/SIMD | Millions of MAC ops/frame needed for >30 FPS real-time |

This is documented as a deliberate architectural decision, not an oversight — a natural extension of this project would be swapping in a lightweight on-device model (e.g., a distilled/quantized keypoint model) if embedded deployment were required.

## Tech stack

- **Python 3.11**
- **OpenCV** — video capture, preprocessing, rendering
- **MediaPipe Holistic** — face mesh + hand landmark extraction
- **NumPy** — vector/matrix math for distance computations

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default webcam
python anxiety_detector.py

# Options
python anxiety_detector.py --camera 0 --width 640 --height 480
python anxiety_detector.py --draw-landmarks     # overlay face mesh + hand skeleton
python anxiety_detector.py --no-preprocess      # disable CLAHE preprocessing
```

Press `q` to quit.

## Limitations & future work

- Thresholds (`FIDGET_THRESHOLD`, `FACE_TOUCH_THRESHOLD`) were calibrated manually for one subject/environment — a more general system would need per-user calibration or a learned classifier trained on labeled behavioral data.
- Single-face tracking only; not designed for multi-person classroom monitoring.
- Rule-based fusion (`OR` logic) is simple by design — could be extended with weighted scoring or a lightweight ML classifier (e.g., logistic regression on the same geometric features) for finer-grained confidence estimates.
- No ground-truth validation dataset was used to compute precision/recall — results are qualitative/demo-based.

## Project structure

```
.
├── anxiety_detector.py   # main pipeline: capture, landmark extraction, rule-based classifier
├── requirements.txt
├── assets/               # demo screenshots/GIFs
└── README.md
```

## Demo

Video demonstration: *(add link here — paste the Google Drive/YouTube link from your submission)*

## Author

Achmad Faiz Abu Bakar — Teknik Komputer, Fakultas Ilmu Komputer, Universitas Brawijaya
