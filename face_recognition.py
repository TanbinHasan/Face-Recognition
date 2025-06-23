# Install required packages with:
# pip install opencv-python numpy deepface

import os
import cv2
import numpy as np
from deepface import DeepFace

# Directory for storing dataset
DATASET_DIR = "Dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# ✅ Create Dataset Function
def create_dataset(name):
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot capture image from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y + h, x:x + w]
            face_path = os.path.join(person_dir, f"{name}_{count}.jpg")
            cv2.imwrite(face_path, face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("Capturing Face", frame)
            cv2.waitKey(200)

            if count >= 50:
                break

        if count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()

# ✅ Train Dataset Function
def train_dataset():
    embedding = {}
    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if os.path.isdir(person_path):
            embedding[person_name] = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                try:
                    vec = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                    embedding[person_name].append(vec)
                except Exception:
                    print("Failed to process image:", img_name)
    return embedding

# ✅ Recognize Face Function
def recognize_face(embeddings):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            try:
                analyse = DeepFace.analyze(face_img, actions=["age", "gender", "emotion"], enforce_detection=False)
                if isinstance(analyse, list):
                    analyse = analyse[0]

                age = analyse.get("age", "N/A")
                gender = analyse.get("gender", "N/A")
                if not isinstance(gender, str) and isinstance(gender, dict):
                    gender = max(gender, key=gender.get)

                emotion = max(analyse.get("emotion", {}), key=analyse["emotion"].get, default="N/A")

                face_embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]

                match = "Unknown"
                max_similarity = -1

                for person_name, embeds in embeddings.items():
                    for embed in embeds:
                        similarity = np.dot(face_embedding, embed) / (
                            np.linalg.norm(face_embedding) * np.linalg.norm(embed)
                        )
                        if similarity > max_similarity:
                            max_similarity = similarity
                            match = person_name

                if max_similarity > 0.7:
                    label = f"{match} ({max_similarity:.2f})"
                else:
                    label = "Unknown"

                display_text = f"{label}, Age: {int(age)}, Gender: {gender}, Emotion: {emotion}"
                cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(255, 255, 255), thickness=2)

            except Exception as e:
                print("Face could not be analyzed or recognized:", str(e))

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ✅ Main Menu
if __name__ == "__main__":
    print("1. Create Face Dataset\n2. Train Face Dataset\n3. Recognize Faces")
    choice = input("Enter your choice: ")

    if choice == "1":
        name = input("Enter the name of the person: ")
        create_dataset(name)

    elif choice == "2":
        embeddings = train_dataset()
        np.save("embeddings.npy", embeddings)
        print("Embeddings saved to embeddings.npy")

    elif choice == "3":
        if os.path.exists("embeddings.npy"):
            embeddings = np.load("embeddings.npy", allow_pickle=True).item()
            recognize_face(embeddings)
        else:
            print("No embeddings file found. Please train the dataset first.")

    else:
        print("Invalid choice.")
