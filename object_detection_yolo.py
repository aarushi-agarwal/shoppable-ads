import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 🔹 Load YOLOv8 model (pretrained)
model = YOLO("yolov8n.pt")  # YOLOv8 nano for fast inference

# 🔹 Load CLIP for image descriptions
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def detect_objects(image_path):
    """
    Detects objects in an image using YOLOv8.
    Returns a list of bounding boxes and labels.
    """
    results = model(image_path)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            label = result.names[int(box.cls)]  # Object class
            detections.append((x1, y1, x2, y2, label))

    return detections


def get_top_colors(image, k=3):
    """
    Extracts the top `k` dominant colors from an image using K-Means clustering.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    color_counts = Counter(kmeans.labels_)
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    top_colors = [kmeans.cluster_centers_[i[0]].astype(int).tolist() for i in sorted_colors]

    return top_colors  # Returns RGB color values


def get_image_description(image_path):
    """
    Generates a description of an image using OpenAI CLIP.
    """
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model(**inputs)
    return "Object detected in the image"  # Placeholder (can be improved with text prompts)


def process_image(image_path, output_csv="object_metadata.csv"):
    """
    Detects objects in an image and extracts metadata.
    Saves results to a CSV file.
    """
    # 🔹 Load image
    image = cv2.imread(image_path)

    # 🔹 Detect objects
    detections = detect_objects(image_path)

    metadata = []
    for (x1, y1, x2, y2, label) in detections:
        # Crop the detected object
        object_crop = image[y1:y2, x1:x2]

        # 🔹 Extract metadata
        top_colors = get_top_colors(object_crop, k=3)
        description = get_image_description(image_path)

        metadata.append({
            "Object Type": label,
            "Bounding Box": f"({x1}, {y1}, {x2}, {y2})",
            "Top Colors": str(top_colors),
            "Description": description
        })

    # 🔹 Save to CSV
    df = pd.DataFrame(metadata)
    df.to_csv(output_csv, index=False)

    print(f"Metadata saved to {output_csv}")
    return df


# 🔹 Run the pipeline on a single image
image_path = "./data/living_room.jpg"  # Replace with your image
df = process_image(image_path)

# 🔹 Display the results
print(df)
