import face_recognition
import os
import pickle

# Path to the folder containing subfolders of known individuals
known_faces_base_path = r"data2"

# Known faces and their names
known_face_encodings = []
known_face_names = []

# Iterate through each subfolder in the base folder
for person_name in os.listdir(known_faces_base_path):
    person_folder = os.path.join(known_faces_base_path, person_name)
    if os.path.isdir(person_folder):
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            try:
                face_image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(face_image)
                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(person_name)
            except Exception as e:
                print(f"Could not process image {image_path}: {e}")

# Save encodings and names to a file
with open('known_faces.pkl', 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print("Encodings saved successfully.")
