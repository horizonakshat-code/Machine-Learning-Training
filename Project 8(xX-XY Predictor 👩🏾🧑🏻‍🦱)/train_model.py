import os
import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Dataset path
dataset_path = dataset_path = "ImageDataset"

images = []
labels = []

# Updated to Male and Female
classes = ["Male", "Female"]

IMG_SIZE = 64

print("Loading and processing images...")

for label, folder in enumerate(classes):
    folder_path = os.path.join(dataset_path, folder)
    
    # Check if folder exists to prevent errors
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found -> {folder_path}")
        continue

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Resize and flatten the image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.flatten()

        images.append(img)
        labels.append(label)

X = np.array(images)
y = np.array(labels)

print(f"Total Training Images: {len(X)}")

# Train the model
print("Training the model...")
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save the model with a new name
joblib.dump(model, "gender_model.pkl")
print("Model Saved Successfully as 'gender_model.pkl'!")