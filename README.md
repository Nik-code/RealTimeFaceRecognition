# LiveFaceRecognizer

A real-time face detection and recognition system using DeepFace and YOLOv8. This project captures live feed from a webcam, detects faces in the feed, and recognizes them based on a predefined database of face images.

## Features
- **Real-time Face Detection:** Uses YOLOv8 for detecting faces in the webcam feed.
- **Face Recognition:** Matches detected faces with a pre-stored database using DeepFace's VGG-Face model.
- **Live Feed:** Continuously processes webcam feed for face detection and recognition.
- **Threshold-based Recognition:** Configurable threshold for recognizing known and unknown faces.

## Installation

### Requirements
- Python 3.7+
- OpenCV
- DeepFace
- NumPy

### Install Dependencies
```bash
pip install opencv-python-headless deepface numpy
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/Nik-code/LiveFaceRecognizer.git
cd LiveFaceRecognizer
```

2. Place images of known persons in separate folders under person directory. For example:
```
person/
  ├── person1/
  │   ├── img1.jpg
  │   └── img2.jpg
  ├── person2/
  │   ├── img1.jpg
  │   └── img2.jpg
```

3. Run the script:
```
python live_face_recognition.py
```

## Code Explanation
### Face Database Preparation

The script processes the images in the `person` directory to create face encodings which are stored in a dictionary for recognition.
```python
for person_name in os.listdir(database_path):
    # Code to process images and create encodings
```

### Face Recognition
The `recognize_face` function compares a detected face with stored encodings and returns the recognized person's name or "Unknown".
```python```
def recognize_face(face_img, face_db, threshold=0.35):
    # Code to recognize faces
```

## Live Webcam Feed
The script captures frames from the webcam, detects faces, and performs recognition in real-time.
```python```
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # Code to detect and recognize faces
```

## Contributing
Feel free to open issues or submit pull requests. Contributions are welcome!

## License
This project is licensed under the MIT License. See the LICENSE file for details.
