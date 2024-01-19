import cv2
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the VGG16 model
model = VGG16(weights=None, include_top=False)  # You can adjust include_top based on your specific needs

# Load your custom model if needed
model = load_model('BrainTumorModel.h5')

# Load and preprocess the image
image_path = '/Users/arshdeepsingh/Documents/Brain_tumor_classification/pred/CT-scan-image-of-brain-tumor.jpg'
img = image.load_img(image_path, target_size=(128, 128))  # Adjust target_size to match VGG16's input size
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)  # Preprocess the image according to VGG16

# Make predictions
result = model.predict(img)
print(result)
