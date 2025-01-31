# teachable-machine-model

# Hand Gesture Recognition Model

This repository contains a machine learning model trained using Google's Teachable Machine to recognize hand gestures representing the numbers **1, 2, and 3**.

## Model Overview
- **Framework:** Teachable Machine (TensorFlow-based)
- **Input Type:** Image (Captured via Webcam or Uploaded)
- **Output Classes:**
  - `One`: Hand gesture representing the number 1 (75 image samples)
  - `Two`: Hand gesture representing the number 2 (75 image samples)
  - `Three`: Hand gesture representing the number 3 (134 image samples)

## How It Works
1. The model takes an image input from the webcam or uploaded images.
2. It processes the image and classifies it into one of the three classes (`One`, `Two`, `Three`).
3. The prediction results are displayed with confidence scores.

## Model Training
- The model was trained using **Teachable Machine**.
- Each class consists of multiple hand gesture images to improve recognition accuracy.
- The trained model can be exported in various formats, including TensorFlow.js, Keras, and TensorFlow Lite.

## Usage
### 1. Running Locally
1. Download the model (`keras_Model.h5`) and label file (`labels.txt`).
2. Load the model using TensorFlow/Keras in Python:

```python
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load class labels
class_names = open("labels.txt", "r").readlines()

# Prepare input image
image = Image.open("hand_gesture.jpg").convert("RGB")
image = ImageOps.fit(image, (224, 224))  # Resize to model input size
image_array = np.asarray(image).astype(np.float32) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Predict
prediction = model.predict(image_array)
predicted_class = class_names[np.argmax(prediction)]
print(f"Predicted Gesture: {predicted_class}")
```

### 2. Using Teachable Machine Web App
1. Open Teachable Machine.
2. Upload or capture an image.
3. View real-time predictions.

## Future Improvements
- Expand dataset to include different lighting conditions and hand positions.
- Train the model with additional gestures.
- Deploy as a web-based application for real-time hand gesture recognition.

