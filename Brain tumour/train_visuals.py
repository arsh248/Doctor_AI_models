import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt  # Import matplotlib for visualization

# Define constants
INPUT_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 7
MODEL_FILE = 'BrainTumorModel.h5'

# Load and preprocess data
image_directory = 'dataset/'
classes = ['no', 'yes']
dataset = []
label = []

for class_index, class_name in enumerate(classes):
    images = os.listdir(os.path.join(image_directory, class_name))
    for image_name in images:
        if image_name.endswith('.jpg'):
            image = cv2.imread(os.path.join(image_directory, class_name, image_name))
            image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
            dataset.append(image)
            label.append(class_index)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    preprocessing_function=keras.applications.vgg16.preprocess_input
)

# Load pre-trained VGG16 model
base_model = VGG16(weights=None, include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
base_model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(MODEL_FILE, monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=(x_test, y_test),
                    steps_per_epoch=len(x_train) / BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=[early_stopping, model_checkpoint])

model.save('final_model.h5')


# Evaluate the model
model = keras.models.load_model(MODEL_FILE)
y_pred = (model.predict(x_test) > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Compute other metrics using the confusion matrix if needed
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


y_train_pred = (model.predict(x_train) > 0.5).astype(int)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

accuracy = accuracy_score(y_test, y_pred)
print("Testing Accuracy: ", accuracy)


# Visualize accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()



