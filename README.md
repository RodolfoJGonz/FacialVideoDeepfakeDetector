# Deepfake Detection with ResNet50 & LSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Active-green)

A deep learning pipeline for detecting deepfake videos by analyzing temporal facial features. This project utilizes a hybrid architecture combining **ResNet50** (for spatial feature extraction) and **LSTM** (for temporal sequence modeling) to classify videos as Real or Fake.

## 🧠 Model Architecture

The model processes a video as a sequence of frames to detect temporal inconsistencies often found in deepfakes.

1.  **Input:** A sequence of 16 face-cropped frames `(Batch, Sequence, Channels, Height, Width)`.
2.  **Feature Extractor (CNN):** A pre-trained **ResNet50** (ImageNet weights) processes each frame independently to extract a 2048-dimensional feature vector. The classification head of ResNet is removed.
3.  **Temporal Modeling (LSTM):** An **LSTM** (Long Short-Term Memory) layer processes the sequence of feature vectors to capture time-dependent patterns (blinking, lip sync, facial movements).
4.  **Classifier:** A fully connected layer takes the last LSTM hidden state and outputs a binary classification: `Real (1)` or `Fake (0)`.

## 📂 Project Structure

```text
├── data/
│   ├── Dataset/           # Raw sorted videos (User provided)
│   ├── labels/            # Generated CSV labels (train/val/test)
│   └── ...
├── models/                # Saved model checkpoints (.pt)
├── cnn_model.py           # (Optional) Baseline CNN training script
├── face_extract.py        # Preprocessing: Extract faces from videos
├── labelling.py           # Preprocessing: Generate CSV labels
├── model.py               # PyTorch Model Class (ResNet50_LSTM)
├── prediction.py          # Inference script for a single video file
├── resize.py              # Utility to resize images
├── test.py                # Evaluate model performance on test set
├── train.py               # Training loop logic
└── vid_frame_index.py     # Main entry point for Training
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RodolfoJGonz/CSCI-4341-DeepfakeDetection.git
    cd deepfake-detection
    ```

2.  **Install dependencies:**
    You will need PyTorch, OpenCV, and RetinaFace.
    ```bash
    pip install torch torchvision opencv-python pandas numpy tqdm pillow scikit-learn retinaface-pytorch
    ```
    or
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage Workflow

### 1. Data Preparation
The pipeline requires extracting faces from videos before training.

1.  **Organize your raw videos** in the following structure:
    ```text
    data/Dataset/
    ├── real/
    │   ├── video1.mp4
    │   └── ...
    ├── fake/
    │   ├── video2.mp4
    │   └── ...
    ```

2.  **Extract Faces:**
    Run `face_extract.py` to crop faces from the videos using RetinaFace.
    ```bash
    python face_extract.py
    ```
    *Note: Verify `face_extract.py` paths match your directory structure before running.*

3.  **Generate Labels:**
    Create the CSV files for training, validation, and testing.
    ```bash
    python labelling.py
    ```
    This will generate `train_labels.csv`, `val_labels.csv`, and `test_labels.csv` in `./data/labels/`.

### 2. Training
To train the ResNet50-LSTM model, run the `vid_frame_index.py` script. This script handles the dataset creation (grouping frames by video ID), sampling 16 frames per video, and running the training loop.

```bash
python vid_frame_index.py
```

* **Configuration:** You can adjust hyperparameters (Epochs, Batch Size, LR) inside `vid_frame_index.py`.
* **Checkpoints:** The best model (highest validation accuracy) is saved to `./models/best.pt`.

### 3. Evaluation
To evaluate the model on the test set and generate a classification report (Precision, Recall, F1-Score):

```bash
python test.py
```

### 4. Inference (Predict on New Video)
To run the model on a raw video file (MP4/MOV) that hasn't been processed yet:

1.  Open `prediction.py`.
2.  Update the `test_video_file` variable with the path to your video.
3.  Run the script:

```bash
python prediction.py
```

What happens during inference?
* The script extracts faces from the video using RetinaFace.
* It samples/pads the faces to a sequence of 16 frames.
* The model outputs the probability of the video being REAL or FAKE.

## 📊 Performance

* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam (`lr=1e-4`)
* **Metric:** Accuracy

## 📝 Notes
* **Sequence Length:** The model expects a fixed sequence length of 16 frames. If a video has fewer frames, it is padded; if it has more, it is evenly sampled.
* **RetinaFace:** This project uses RetinaFace for high-accuracy face detection. A GPU is recommended for preprocessing large datasets.

## 🤝 Contributing
Feel free to open issues or submit pull requests if you have suggestions for improving the architecture or data processing pipeline.


