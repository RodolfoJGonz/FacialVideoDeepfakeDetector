import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from retinaface import RetinaFace
from model import ResNet50_LSTM # Assuming model.py is accessible
import torch.nn.functional as F


# --- 1. FACE EXTRACTION & RESIZING ---

def detect_and_crop_face(frame, frame_idx, output_dir, size=(224, 224)):
    """Detects, crops, and resizes the largest face in a frame."""
    detections = RetinaFace.detect_faces(frame)
    if not detections:
        return None

    # Find the largest detected face
    largest_face = None
    max_area = 0
    for face in detections.values():
        x1, y1, x2, y2 = face["facial_area"]
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            largest_face = face

    if largest_face:
        x1, y1, x2, y2 = largest_face["facial_area"]
        cropped_face = frame[y1:y2, x1:x2]
        resized = cv2.resize(cropped_face, size)
        
        # Save temporarily (similar to the original workflow)
        output_path = os.path.join(output_dir, f"face_{frame_idx:04d}.jpg")
        cv2.imwrite(output_path, resized)
        return output_path
    return None

def extract_faces_from_video(video_path, output_dir, frame_skip=32):
    """Processes a video to extract, crop, and save face images."""
    print(f"Processing video: {video_path}")
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    frame_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_skip == 0:
            path = detect_and_crop_face(frame, frame_num, output_dir)
            if path:
                frame_paths.append(path)

        frame_num += 1

    cap.release()
    print(f"Extracted {len(frame_paths)} face frames.")
    return frame_paths

# --- 2. SEQUENCE SAMPLING LOGIC (From DeepfakeSequenceDataset) ---

def sample_or_pad(frames, seq_len=16):
    """Return exactly seq_len frames by sampling or padding."""
    n = len(frames)
    if n == seq_len:
        return frames

    if n > seq_len:
        # Evenly pick seq_len frames
        idxs = np.linspace(0, n-1, seq_len).astype(int)
        return [frames[i] for i in idxs]

    # n < seq_len → pad last frame
    last = frames[-1]
    padded = frames + [last] * (seq_len - n)
    return padded

# --- 3. INFERENCE ---

def predict_video(video_path, model, device="cuda", seq_len=16):
    temp_dir = "./temp_video_faces"
    
    # Step 1: Extract, crop, and resize faces
    frame_paths = extract_faces_from_video(video_path, temp_dir)
    
    if not frame_paths:
        print("Error: No faces extracted from video.")
        return None

    # Step 2: Sample/Pad sequence
    selected_frames = sample_or_pad(frame_paths, seq_len=seq_len)

    # Step 3: Apply transformation (used in vid_frame_index.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load images and apply transformation
    imgs = []
    for p in selected_frames:
        img = Image.open(p).convert("RGB")
        img_tensor = transform(img)
        imgs.append(img_tensor)

    # Stack into expected tensor shape: [1, T, C, H, W]
    # [1: Batch size, T: Time/Sequence length (16), C: Channels (3), H, W: Height/Width (224)]
    video_tensor = torch.stack(imgs).unsqueeze(0).to(device)

    # Step 4: Inference
    model.eval()
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)
        
        # Get prediction
        pred_class = probabilities.argmax().item()
        
        # Cleanup temporary files (Optional, but good practice)
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
        
        return pred_class, probabilities.cpu().numpy()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # ⚠️ EDIT THIS PATH TO YOUR VIDEO
    test_video_file = "/Users/nathanieldeleon/Desktop/currentVid.mov" 
    
    if not os.path.exists(test_video_file):
        print(f"Error: Video file not found at {test_video_file}. Please update the path.")
        exit()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet50_LSTM()
    
    # Load the best weights saved during training
    try:
        model.load_state_dict(torch.load("./models/best.pt", map_location=device))
        model.to(device)
    except FileNotFoundError:
        print("Error: Model weights './models/best.pt' not found. Ensure you have trained the model.")
        exit()

    # Make prediction
    result = predict_video(test_video_file, model, device=device)

    if result is not None:
        pred_class, probs = result
        label_map = {0: "FAKE", 1: "REAL"}
        
        print("\n--- Prediction Result ---")
        print(f"Predicted Class: **{label_map[pred_class]}**")
        print(f"Probability (REAL): {probs[1]:.4f}")
        print(f"Probability (FAKE): {probs[0]:.4f}")
    else:
        print("\n--- Prediction Failed ---")