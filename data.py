import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the training and validation data
(train_images, train_labels), (val_images, val_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
train_images = train_images / 255.0
val_images = val_images / 255.0

# Reshape the images to have a single color channel
train_images = train_images.reshape(-1, 28, 28, 1)
val_images = val_images.reshape(-1, 28, 28, 1)

# Resize the images to a larger size
train_images = tf.image.resize(train_images, (128, 128))
val_images = tf.image.resize(val_images, (128, 128))

# Convert the images to have 3 color channels
train_images = tf.repeat(train_images, 3, axis=-1)
val_images = tf.repeat(val_images, 3, axis=-1)

# Convert labels to one-hot encoded format
num_classes = 10
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
val_labels = tf.keras.utils.to_categorical(val_labels, num_classes)

# Load the pre-trained model from TensorFlow Hub
module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/4"
hub_layer = hub.KerasLayer(module_url, trainable=False)

# Build the transfer learning model
model = tf.keras.Sequential([
    hub_layer,
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))

# Get the training and validation accuracy from the history object
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Create the plot for performance comparison
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Training Accuracy')
plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Performance Comparison')
plt.legend()

# Add noise to the validation images
noisy_val_images = val_images + np.random.normal(loc=0.0, scale=0.1, size=val_images.shape)
noisy_val_images = np.clip(noisy_val_images, 0.0, 1.0)

# Evaluate the model on the noisy validation images
noise_val_loss, noise_val_accuracy = model.evaluate(noisy_val_images, val_labels)

# Create the plot for robustness to noise
plt.figure(figsize=(8, 6))
plt.bar(['Noisy Validation'], [noise_val_accuracy], label='Noisy Validation')
plt.ylim([0, 1])
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('Robustness to Noise')
plt.legend()

# Limit the labeled data to simulate limited labeled data scenario
limit = 5000
limited_train_images = train_images[:limit]
limited_train_labels = train_labels[:limit]

# Train the model with limited labeled data
limited_model = tf.keras.Sequential([
    hub_layer,
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the limited model
limited_model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

# Train the limited model
limited_history = limited_model.fit(limited_train_images, limited_train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))

# Get the training and validation accuracy from the limited history object
limited_train_accuracy = limited_history.history['accuracy']
limited_val_accuracy = limited_history.history['val_accuracy']

# Create the plot for limited labeled data
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(limited_train_accuracy) + 1), limited_train_accuracy, label='Limited Training Accuracy')
plt.plot(range(1, len(limited_val_accuracy) + 1), limited_val_accuracy, label='Limited Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Limited Labeled Data')
plt.legend()
plt.tight_layout()
plt.show()