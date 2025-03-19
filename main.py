import cv2
import numpy as np
import os

# Create a directory for storing face images if not exists
if not os.path.exists("faces"):
    os.makedirs("faces")

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to capture images for training
def capture_images(name):
    cap = cv2.VideoCapture(0)
    count = 0
    while count < 50:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(f"faces/{name}_{count}.jpg", face)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to train the recognizer
def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    label_dict = {}
    label_id = 0
    for file in os.listdir("faces"):
        path = os.path.join("faces", file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        name = file.split("_")[0]
        if name not in label_dict:
            label_dict[name] = label_id
            label_id += 1
        faces.append(np.array(img, dtype=np.uint8))
        labels.append(label_dict[name])
    recognizer.train(faces, np.array(labels))
    recognizer.save("face_model.yml")
    return label_dict

# Function to recognize faces
def recognize_faces(label_dict):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_model.yml")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)
            name = [k for k, v in label_dict.items() if v == label][0]
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Main flow
if __name__ == "__main__":
    name = input("Enter your name: ")
    capture_images(name)
    label_dict = train_recognizer()
    recognize_faces(label_dict)