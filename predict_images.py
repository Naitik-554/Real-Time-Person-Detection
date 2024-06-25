import cv2
import torch
from ultralytics import YOLO
import face_recognition
import pickle
import os

# Load YOLOv5 model
model = YOLO(r"runs\detect\train-5epochs\weights\best.pt")

# Load known faces and names from file
with open('known_faces.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

# List of image paths
image_paths = [
    r"data2\modi\Prime_Minister_Shri_Narendra_Modi_in_New_Delhi_on_August_08_2019_cropped.jpg",
    r"C:\Users\NAITIK\Desktop\Person Recognition\data2\modi\aug_1_gettyimages-496809772-1024x1024.jpg",
    r"C:\Users\NAITIK\Desktop\Person Recognition\data2\modi\aug_1_Narendra-Modi-PNG-Transparent-Image.png",
    r"C:\Users\NAITIK\Desktop\Person Recognition\data2\modi\gettyimages-496809772-1024x1024.jpg",
    r"C:\Users\NAITIK\Desktop\Person Recognition\data2\modi\aug_4_modi_tensed.jpeg",
    r"C:\Users\NAITIK\Desktop\Person Recognition\data2\modi\Narendra-Modi-10-1.jpg",
    r"C:\Users\NAITIK\Desktop\Person Recognition\data2\modi\Narendra-Modi (1).jpg",
    r"C:\Users\NAITIK\Desktop\Person Recognition\data2\modi\gettyimages-1669190297-1024x1024.jpg",
    r"C:\Users\NAITIK\Desktop\Person Recognition\data2\modi\gettyimages-1531836496-1024x1024.jpg",
    r"C:\Users\NAITIK\Desktop\Person Recognition\data2\modi\gettyimages-1500649380-1024x1024.jpg",
    r"C:\Users\NAITIK\Desktop\Person Recognition\data2\modi\modi_tensed.jpeg"
]

# Output directory for processed images
output_dir = "processed_images"
os.makedirs(output_dir, exist_ok=True)

# Define a higher tolerance for face comparison
tolerance = 0.8

for image_path in image_paths:
    # Load the image
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    # Use YOLO to detect people in the image
    results = model(image)

    # Process the detection results
    for i in range(len(results[0].boxes.cls)):
        class_id = int(results[0].boxes.cls[i].item())
        confidence = results[0].boxes.conf[i].item()
        bbox = results[0].boxes.data[i][:4].tolist()
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Extract the detected person's face
        face_image = image[y1:y2, x1:x2]

        # Convert the image to RGB for face_recognition
        rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Find face encodings in the detected face
        face_encodings = face_recognition.face_encodings(rgb_face_image)

        name = "Unknown"
        if face_encodings:
            # Compare detected face with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0], tolerance=tolerance)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
            best_match_index = face_distances.argmin()

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Draw rectangle and display class, confidence, and name
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{name} ({confidence:.2f})"

        # Calculate text size to ensure it fits within the image
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_x = x1
        text_y = y1 - 10

        # Adjust position if text goes out of image bounds
        if text_y - text_height < 0:
            text_y = y1 + text_height + 10
        if text_x + text_width > image_width:
            text_x = image_width - text_width - 10

        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the resulting image with bounding boxes and names
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)

print("Predictions saved successfully.")