import cv2
import os

# video = yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]" --merge-output-format mp4 -o "./%(title)s.mp4" "https://www.youtube.com/watch?v=mVhiMl8Gew4"

video_path = "./data/house_tour.mp4"

output_folder = "/data/frames"
os.makedirs(output_folder, exist_ok=True)

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file {video_path} not found")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file {video_path}")

frame_rate = 5  # Process every 5th frame
frame_count = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_rate == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
        
        frame_count += 1
finally:
    cap.release()
    cv2.destroyAllWindows()

print(f"Frames extracted and saved in {output_folder}")