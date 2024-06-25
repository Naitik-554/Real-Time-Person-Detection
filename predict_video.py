import cv2
import torch
from ultralytics import YOLO
import face_recognition
import os

# Load YOLOv8 model
model = YOLO(r"runs\detect\train-5epochs\weights\best.pt")

# Known faces and their names
known_face_encodings = []
known_face_names = []

# Add known faces
known_faces_path = r"data2/modi_aug"  # Folder containing images of known individuals

# Encode known faces
for file_name in os.listdir(known_faces_path):
    if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".jpeg"):
        face_path = os.path.join(known_faces_path, file_name)
        face_image = face_recognition.load_image_file(face_path)
        face_encodings = face_recognition.face_encodings(face_image)
        if face_encodings:
            face_encoding = face_encodings[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append("Narendra Modi")

# Load the video
video_path = r"videos\uae-modi.mp4"
output_path = r"videos/uae_modi_output.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def process_frame(frame):
    # Use YOLO to detect people in the frame
    results = model(frame)

    # Process the detection results
    for i in range(len(results[0].boxes.cls)):
        class_id = int(results[0].boxes.cls[i].item())
        confidence = results[0].boxes.conf[i].item()
        bbox = results[0].boxes.data[i][:4].tolist()
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Extract the detected person's face
        face_image = frame[y1:y2, x1:x2]
        
        # Convert the image to RGB for face_recognition
        rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Find face encodings in the detected face
        face_encodings = face_recognition.face_encodings(rgb_face_image)

        name = "Unknown"
        if face_encodings:
            # Compare detected face with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0], tolerance=0.6)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
            best_match_index = face_distances.argmin()
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Draw rectangle and display class, confidence, and name
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{name} ({confidence:.2f})"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)
    out.write(processed_frame)

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
