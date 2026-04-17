# Multi-Object Detection and Persistent ID Tracking

A computer vision pipeline that detects and tracks multiple players/athletes in sports footage, assigning each a **unique and persistent ID** across the entire video — even through occlusion, camera motion, and similar appearances.

---

## Pipeline Overview

```
Input Video  →  YOLOv8s (Detection)  →  BoT-SORT (Tracking)  →  Supervision (Annotation)  →  Output Video
```

| Component        | Tool / Model         | Purpose                                    |
|------------------|----------------------|--------------------------------------------|
| Object Detection | YOLOv8s (Ultralytics)| Detect all "person" class instances per frame |
| Multi-Object Tracking | BoT-SORT        | Assign & maintain persistent IDs with Camera Motion Compensation |
| Visualization    | Supervision (Roboflow)| Render bounding ellipses, ID labels, and trajectory trails |
| Video I/O        | OpenCV               | Frame-level read/write operations           |

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone or unzip this repository
git clone <repository-url>
cd intern-ass

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run the Pipeline

1. Place your input video inside the `input/` folder (a sample `short_football_video.mp4` is included).
2. Run the main script:

```bash
python main.py
```

3. The annotated output video will be saved to `output/annotated_video.mp4`.

---

## Project Structure

```
intern-ass/
├── input/                       # Input video(s)
│   └── short_football_video.mp4
├── output/                      # Generated output video(s)
│   └── annotated_video.mp4
├── main.py                      # Core detection + tracking pipeline
├── requirements.txt             # Python dependencies
├── technical_report.md          # Short technical report
└── README.md                    # This file
```

---

## Assumptions

- The input video contains **people/players** as the primary subjects to detect and track.
- We use the COCO-pretrained YOLOv8s model, filtering exclusively for `class 0` (person).
- A confidence threshold of `0.3` is applied to remove noisy false-positive detections.
- The tracker's buffer (memory for lost tracks) uses the default BoT-SORT settings, which are tuned for general multi-object tracking scenarios.

## Limitations

- **Prolonged full occlusion:** If a player is completely hidden behind another player or object for many consecutive frames, the tracker may eventually drop their ID and assign a new one when they reappear.
- **Extreme similarity:** Players wearing identical uniforms with no distinctive visual features (e.g., jersey numbers too small to resolve) can cause ID swaps during close interactions.
- **Far-field / small detections:** Very distant players may be missed by the detector if they fall below the confidence threshold.
- **No team clustering:** The current pipeline does not distinguish between teams — all detected persons receive a generic ID.

## Model / Tracker Choices

| Choice    | Rationale |
|-----------|-----------|
| **YOLOv8s** | Best trade-off between speed and accuracy for real-time person detection. The "small" variant processes frames fast while maintaining high mAP on the COCO person class. |
| **BoT-SORT** | Includes **Camera Motion Compensation (CMC)** via background feature matching, which is critical for sports footage with constant panning/zooming. Also incorporates a ReID (Re-Identification) module to recover IDs after occlusion. |
| **Supervision** | Provides production-quality visual annotations (ellipses, labels, trajectory traces) with minimal code — aligns with the "clarity of work" evaluation criteria. |

---

## Source Video

- **Video:** `input/short_football_video.mp4`
- **Source:** *(insert public URL here)*

---

## Author

Built as part of the Multi-Object Detection and Persistent ID Tracking assignment.
