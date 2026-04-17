import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
from scipy.optimize import linear_sum_assignment
from collections import deque
from ultralytics import YOLO

# =============================================================================
# RE-ID FEATURE EXTRACTOR
# =============================================================================
class ReIDExtractor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = mobilenet_v2(weights='IMAGENET1K_V1').eval().to(self.device)
        self.model.classifier = torch.nn.Identity()
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def get_features(self, frame, bboxes):
        features = []
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                features.append(np.zeros(1280))
                continue
            tensor = self.preprocess(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
            feat = F.normalize(self.model(tensor), p=2, dim=1)
            features.append(feat.cpu().numpy().flatten())
        return np.array(features) if features else np.empty((0, 1280))

# =============================================================================
# PERSISTENT TRACKER ENGINE
# =============================================================================
class Track:
    _id_count = 1
    def __init__(self, bbox, feat):
        self.id = Track._id_count
        Track._id_count += 1
        self.bbox = bbox
        self.features = deque([feat], maxlen=50)
        self.time_since_update = 0

    def update(self, bbox, feat):
        self.bbox = bbox
        self.features.append(feat)
        self.time_since_update = 0

    @property
    def feature_synopsis(self):
        return np.mean(self.features, axis=0)

class FootballTracker:
    def __init__(self, max_lost_frames=90):
        self.tracks = []
        self.max_lost = max_lost_frames

    def _calculate_iou(self, boxA, boxB):
        xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / float(areaA + areaB - inter + 1e-6)

    def step(self, bboxes, feats):
        for t in self.tracks:
            t.time_since_update += 1

        unmatched_dets = list(range(len(bboxes)))
        unmatched_tracks = list(range(len(self.tracks)))

        if unmatched_tracks and unmatched_dets:
            cost_matrix = np.zeros((len(unmatched_tracks), len(unmatched_dets)))
            for i, t_idx in enumerate(unmatched_tracks):
                for j, d_idx in enumerate(unmatched_dets):
                    app_dist = 1.0 - np.dot(self.tracks[t_idx].feature_synopsis, feats[d_idx])
                    iou_dist = 1.0 - self._calculate_iou(self.tracks[t_idx].bbox, bboxes[d_idx])
                    cost_matrix[i, j] = 0.8 * app_dist + 0.2 * iou_dist

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_t, assigned_d = [], []
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 0.4:
                    self.tracks[unmatched_tracks[r]].update(bboxes[unmatched_dets[c]], feats[unmatched_dets[c]])
                    assigned_t.append(unmatched_tracks[r])
                    assigned_d.append(unmatched_dets[c])
            
            unmatched_tracks = [i for i in unmatched_tracks if i not in assigned_t]
            unmatched_dets = [i for i in unmatched_dets if i not in assigned_d]

        for d_idx in unmatched_dets:
            self.tracks.append(Track(bboxes[d_idx], feats[d_idx]))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost]
        return [(t.id, t.bbox) for t in self.tracks if t.time_since_update == 0]

# =============================================================================
# MAIN EXECUTION (SAVE ONLY)
# =============================================================================
def process_and_save(input_video, output_folder="output"):
    # 1. Prepare Output Directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_path = os.path.join(output_folder, "annotated_football.mp4")

    # 2. Initialize Models
    detector = YOLO("yolov8m.pt")
    reid = ReIDExtractor()
    tracker = FootballTracker(max_lost_frames=90)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open {input_video}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"Starting processing: {total_frames} frames...")

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = detector(frame, classes=[0], conf=0.4, verbose=False)[0]
        bboxes = results.boxes.xyxy.cpu().numpy()

        if len(bboxes) > 0:
            feats = reid.get_features(frame, bboxes)
            online_targets = tracker.step(bboxes, feats)

            for tid, box in online_targets:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{tid}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1
        
        # Console feedback since there is no live window
        if frame_count % 50 == 0:
            print(f"Progress: {frame_count}/{total_frames} frames processed.")

    cap.release()
    out.release()
    print(f"Success! Video saved to: {output_path}")

if __name__ == "__main__":
    # Replace with your actual video filename
    process_and_save("C:\Machine Learning\intern-ass\input\input.mp4")