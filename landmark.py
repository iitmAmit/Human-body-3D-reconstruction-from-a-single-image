import cv2
import numpy as np

# Load pre-trained model
model_path = 'lbfmodel.yaml'
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(model_path)

# Load image
image_path = 'boy-3648740_640.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Example landmark indices (You can choose indices you need for the 5 landmarks)
landmark_indices = [36, 45, 30, 48, 54]  # Example indices

# Detect landmarks
success, landmarks = facemark.fit(gray, faces)

if success:
    for (face, landmark) in zip(faces, landmarks):
        # Extract the selected landmarks
        landmarks_array = np.array([landmark[0][i] for i in landmark_indices])
        print("Landmarks (5x2):")
        print(landmarks_array)

        # Optionally, visualize the landmarks on the image
        for (x, y) in landmarks_array:
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)

# Display result
cv2.imshow('Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()





