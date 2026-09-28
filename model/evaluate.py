import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load model
model = tf.keras.models.load_model("oral_model.h5")

# Data generator (NO augmentation here)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load validation data
val_data = datagen.flow_from_directory(
    '../dataset/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Get predictions
preds = model.predict(val_data)
y_pred = (preds > 0.5).astype(int).reshape(-1)

# True labels
y_true = val_data.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Cancer']))