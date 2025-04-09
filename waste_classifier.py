
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image
import matplotlib.pyplot as plt

# Feature extractor: convert image to grayscale and resize
def extract_features(image_path, size=(50, 50)):
    try:
        img = Image.open(image_path).convert('L')  # grayscale
        img = img.resize(size)
        return np.array(img).flatten()
    except:
        return None

# Define your dataset path
dataset_path = '/storage/emulated/0/Download/dataset'  # adjust this path if needed
classes = os.listdir(dataset_path)

X, y = [], []

# Load data
for label in classes:
    class_dir = os.path.join(dataset_path, label)
    if not os.path.isdir(class_dir):
        continue
    for file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, file)
        features = extract_features(img_path)
        if features is not None:
            X.append(features)
            y.append(label)

# Train/test split
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
