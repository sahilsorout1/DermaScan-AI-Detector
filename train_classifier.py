import os

# --- MAC SAFETY FIXES (Must be at the very top) ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import tensorflow as tf
import tensorflow as tf
from keras import layers, models
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# --- CONFIGURATION ---
BATCH_SIZE = 32
IMG_SIZE = (224,224)
EPOCHS = 20

# 1. Setup Paths
base_dir = 'dataset'
image_dir = os.path.join(base_dir, 'images')
csv_path = 'dataset/HAM10000_metadata.csv'


print("Loading Data Information...")
df = pd.read_csv(csv_path)

# 2. Fix Column Names
df['path'] = df['image'] + '.jpg' 
label_columns = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# 3. Create Generators (Standard Scaling)
datagen = ImageDataGenerator(
    rescale=1./255,       # Using standard scaling
    rotation_range=20,
    horizontal_flip=True,
    validation_split=0.2
)

print("Setting up Training Generator...")
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col='path',
    y_col=label_columns,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw',
    subset='training'
)

print("Setting up Validation Generator...")
val_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col='path',
    y_col=label_columns,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw',
    subset='validation'
)

# 4. Build Model
print("Downloading the AI Brain...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# --- THIS IS THE CRITICAL CHANGE ---
base_model.trainable = True  # Unfreezing so it can learn Skin Cancer specifically

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

# Using a lower learning rate (1e-4) because the model is unfrozen
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# 5. Train
print(f"Starting Training for {EPOCHS} Epochs...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# 6. Save
if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/skin_cancer_model.h5')
print("\n✅ SUCCESS! Model saved to 'models/skin_cancer_model.h5'")
