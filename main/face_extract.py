from retinaface import RetinaFace
import cv2, os, glob

def detect_and_crop_face(frame, frame_idx, output_dir):
    detections = RetinaFace.detect_faces(frame)
    if not detections:
        print(f"No face detected in frame {frame_idx}")
        return

    # Find the largest detected face (by bounding box area)
    largest_face = None
    max_area = 0

    for face in detections.values():
        x1, y1, x2, y2 = face["facial_area"]
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            largest_face = face

    # Crop the largest face
    if largest_face:
        x1, y1, x2, y2 = largest_face["facial_area"]
        cropped_face = frame[y1:y2, x1:x2]
        output_path = os.path.join(output_dir, f"face_{frame_idx:04d}.jpg")
        cv2.imwrite(output_path, cropped_face)


def extract_faces_from_video(video_path, output_root, frame_skip=32):
    # Name output folder by video name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_root, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame
        if frame_num % frame_skip == 0:
            detect_and_crop_face(frame, frame_num, video_output_dir)

        frame_num += 1

    cap.release()
    print(f"Processed video {video_name}, total frames: {frame_num}")


# Example usage
# Iterate through videos in video path


videos_path = "./data/original_sequences/youtube/c23/videos"
output_root = "./data/original_sequences/youtube/c23/vid_faces/"

#videos_path = "./data/manipulated_sequences/Deepfakes/c23/videos"
#output_root = "./data/manipulated_sequences/Deepfakes/c23/vid_faces/"

for vid in glob.glob(f"{videos_path}/*.mp4"):
    extract_faces_from_video(vid,output_root, frame_skip=32)
#extract_faces_from_video(video_path, output_root, frame_skip=32)
