import cv2
import numpy as np

from adas.perception.annotate import draw_lanes
from adas.perception.lanes import LaneDetector

cap = cv2.VideoCapture(r"data\videos\drive1.mp4")
for _ in range(30):                      # go to frame 30
    ok, frame = cap.read()
cap.release()

det = LaneDetector()
h, w = frame.shape[:2]
poly = det._roi_polygon(w, h)

# Steps 1-3
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, det.canny_low, det.canny_high)

# Step 4
mask = np.zeros_like(edges)
cv2.fillPoly(mask, poly, 255)
masked = cv2.bitwise_and(edges, mask)
roi_view = frame.copy()
cv2.polylines(roi_view, poly, True, (0, 255, 255), 3)

# Step 5
segments = cv2.HoughLinesP(masked, 1, np.pi / 180, det.hough_threshold,
                           minLineLength=det.min_line_length,
                           maxLineGap=det.max_line_gap)
segments = np.empty((0, 4)) if segments is None else segments.reshape(-1, 4)

hough_view = frame.copy()
sorted_view = frame.copy()
n_flat = n_left = n_right = n_wrong = 0
split = det.split_x * w
for x1, y1, x2, y2 in segments.astype(float):
    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
    cv2.line(hough_view, p1, p2, (0, 255, 0), 3)           # every segment: green
    slope = (y2 - y1) / (x2 - x1) if x1 != x2 else 0.0
    if x1 == x2 or abs(slope) < det.min_abs_slope:
        n_flat += 1
        color = (128, 128, 128)                              # too flat: gray
    elif slope < 0 and max(x1, x2) < split:
        n_left += 1
        color = (255, 0, 0)                                  # left candidate: blue
    elif slope > 0 and min(x1, x2) > split:
        n_right += 1
        color = (0, 0, 255)                                  # right candidate: red
    else:
        n_wrong += 1
        color = (0, 165, 255)                                # wrong side/sign: orange
    cv2.line(sorted_view, p1, p2, color, 3)

# Steps 7-8
final = frame.copy()
est = det.detect(frame)
draw_lanes(final, est)

for name, img in [("s1_gray", gray), ("s2_blur", blur), ("s3_edges", edges),
                  ("s4_roi", roi_view), ("s5_masked_edges", masked),
                  ("s6_hough_segments", hough_view), ("s7_sorted", sorted_view),
                  ("s8_final", final)]:
    cv2.imwrite(f"outputs/{name}.jpg", img)

print("Edge pixels before mask:", int(np.count_nonzero(edges)))
print("Edge pixels after mask :", int(np.count_nonzero(masked)))
print("Hough segments         :", len(segments))
print(f"  too flat: {n_flat} | left: {n_left} | right: {n_right} | wrong side/sign: {n_wrong}")
print(est)