import cv2
from deepface import DeepFace
import time
import numpy as np
import os

# Set the path to the database of known persons
database_path = r'--your directory for faces--'

# Create a dictionary to store face encodings
face_db = {}

print("Starting to process the face database...")
start_time = time.time()

# Process each person's images in the database
for person_name in os.listdir(database_path):
    person_path = os.path.join(database_path, person_name)
    if os.path.isdir(person_path):
        print(f"Processing {person_name}...")
        encodings = []
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            print(f"Processing image {img_name}...")
            try:
                # Detect and encode the face
                face_objs = DeepFace.extract_faces(img_path, detector_backend='yolov8', enforce_detection=False)
                for face in face_objs:
                    encoding = DeepFace.represent(face['face'], model_name='VGG-Face', enforce_detection=False)[0][
                        "embedding"]
                    encodings.append(encoding)
            except Exception as e:
                print(f"Error processing image {img_name}: {e}")
        # Store encodings in the dictionary
        face_db[person_name] = encodings
        print(f"Finished processing {person_name}")

print(f"Finished processing all faces. Total persons processed: {len(face_db)}")


def recognize_face(face_img, face_db, threshold=0.35):
    """
    Recognize the face from the face database.

    Args:
        face_img (np.array): Image of the face to be recognized.
        face_db (dict): Dictionary of known face encodings.
        threshold (float): Distance threshold for face recognition.

    Returns:
        str: Name of the recognized person or "Unknown".
    """
    try:
        encoding = DeepFace.represent(face_img, model_name='VGG-Face', enforce_detection=False)[0]["embedding"]
        distances = {}
        for person_name, encodings in face_db.items():
            dist = np.min([np.linalg.norm(np.array(encoding) - np.array(enc)) for enc in encodings])
            distances[person_name] = dist

        if distances:
            min_dist = min(distances.values())
            print(min(distances, key=distances.get), min_dist)
            if min_dist > threshold:
                return min(distances, key=distances.get)
            else:
                return "Unknown"
        else:
            return "Unknown"
    except Exception as e:
        print(f"Error in face recognition: {e}")
        return "Unknown"


# Initialize webcam capture
cap = cv2.VideoCapture(0)
cv2.startWindowThread()

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

last_recognition_time = 0
recognition_interval = 0  # seconds
recognized_faces = {}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    current_time = time.time()

    try:
        # Use DeepFace to detect faces with YOLOv8
        faces = DeepFace.extract_faces(frame, detector_backend='yolov8', enforce_detection=False)

        # Draw rectangles around the detected faces
        for face in faces:
            x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], \
            face['facial_area']['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

            # Extract face for recognition
            face_img = frame[y:y + h, x:x + w]

            # Perform face recognition
            if current_time - last_recognition_time >= recognition_interval:
                name = recognize_face(face_img, face_db)
                recognized_faces[(x, y, w, h)] = name
                last_recognition_time = current_time

            # Display the name
            if (x, y, w, h) in recognized_faces:
                name = recognized_faces[(x, y, w, h)]
                cv2.putText(frame, f'Name: {name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    except Exception as e:
        print(f"Error in face detection: {e}")

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()
