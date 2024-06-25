import os
from ultralytics import YOLO
import cv2

# Load the model
model_path = os.path.join('.', 'runs', 'detect', 'train7', 'weights', 'best.pt')
model = YOLO(model_path)  # load a custom model

threshold = 0.5

# Open the webcam
cap = cv2.VideoCapture(0)  # '0' is the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get the video frame dimensions
ret, frame = cap.read()
H, W, _ = frame.shape

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('YOLO Live Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
