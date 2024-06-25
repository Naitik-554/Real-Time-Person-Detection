from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # Load a pretrained model (recommended for training)

# Train the model on GPU
results = model.train(data=r"C:\Users\NAITIK\Desktop\Person Recognition\data\data.yaml", epochs=1)  # Adjust the number of epochs as needed