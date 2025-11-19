
import cv2, glob, os

def resize_faces(input_dir, output_dir, folder_name, size=(224,224)):
    os.makedirs(output_dir, exist_ok=True)
    for img_path in glob.glob(f"{input_dir}/face_*.jpg"):
        img = cv2.imread(img_path)
        resized = cv2.resize(img, size)
        filename = os.path.basename(img_path)
        file = os.path.join(output_dir, folder_name)
        os.makedirs(file, exist_ok=True)
        cv2.imwrite(os.path.join(file, filename), resized)


input_dir = "./data/original_sequences/youtube/c23/vid_faces"
output_dir = "./data/original_sequences/youtube/c23/real"

#input_dir = "./data/manipulated_sequences/Deepfakes/c23/vid_faces"
#output_dir = "./data/manipulated_sequences/Deepfakes/c23/fake"

#for video in glob.glob(f"{input_dir}/*/"):
#    dir = input_dir.split("/");
#    print(dir)
#    resize_faces(video, output_dir, dir[-1])

for subdir, dir, files in os.walk(input_dir):
    for folder in dir:
        working_dir = os.path.join(input_dir, folder)
        resize_faces(working_dir, output_dir, folder)
