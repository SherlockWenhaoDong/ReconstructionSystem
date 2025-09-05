import cv2
import os

# ========== Parametres ==========
video_path = "/home/dongwenhao/SurgicalRecon/Stereo_Skilltest/StereoRecordings/R_clip2.mp4"
out_dir = "/home/dongwenhao/SurgicalRecon/frames/R_clip2"
os.makedirs(out_dir, exist_ok=True)

# old resolutions
orig_h, orig_w = 1008, 1264
# new resolutions
crop_h, crop_w = 840, 1250

# get the central area
x1 = (orig_w - crop_w) // 2  # 120
y1 = (orig_h - crop_h) // 2  # 24
x2 = x1 + crop_w             # 1144
y2 = y1 + crop_h             # 984

# ========== get frames ==========
cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cropped = frame[y1:y2, x1:x2]

    save_path = os.path.join(out_dir, f"frame_{frame_id:05d}.png")
    cv2.imwrite(save_path, cropped)
    print(f'Finish writing frame-{frame_id:05d}!')

    frame_id += 1

cap.release()
