import tensorflow as tf
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Scripts needed to train the model
from data_loader import load_datasets
from model_architecture import create_model

# ===================================================================
# HYPERPARAMETER AND CONSTANTS DEFINITION
# ===================================================================
BASE_DATA_DIR = 'C:\\Users\\Programmer\\processed'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# ===================================================================
# UPLOAD THE DATA
# ===================================================================
train_ds, val_ds, test_ds, class_names = load_datasets(
    base_dir=BASE_DATA_DIR,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ===================================================================
# EXPERIENCE CONFIGURATION
# ===================================================================
EXPERIMENTS_BASE_DIR = 'C:\\Users\\Miguel António\\Desktop\\PORTFOLIO\\image_classifier\\experiments'

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_dir = os.path.join(EXPERIMENTS_BASE_DIR, timestamp)
os.makedirs(experiment_dir, exist_ok=True)
print(f"\nExperience initialized - Results will be saved in: {experiment_dir}")

model_checkpoint_path = os.path.join(experiment_dir, "best_model.keras")
# ===================================================================
# CONSTRUCTION AND COMPILATION OF THE MODEL
# ===================================================================
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
num_classes = len(class_names)

model = create_model(input_shape=input_shape, num_classes=num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ===================================================================
# CALLBACKS AND TRAINING CONFIGURATION
# ===================================================================
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

print(f"\nInicialize training for {EPOCHS} epochs")
history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, 
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# ===================================================================
# EVALUATION AND VISUALIZATION
# ===================================================================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy per epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss per epoch')
plt.legend()

plot_path = os.path.join(experiment_dir, "training_plots.png")
plt.savefig(plot_path)
print(f"Training plots saved in: {plot_path}")

# Evaluation
print("\nFinal evaluation on test dataset for with the best weights:")
loss, accuracy = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")