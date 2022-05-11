# Databricks notebook source
# MAGIC %md
# MAGIC # Convolutional Neural Networks (CNN)

# COMMAND ----------

# MAGIC %md
# MAGIC - Step 1: Load data
# MAGIC     - Fashion MNIST: Like MNIST but with clothing images, 28x28 grayscale images, labels: t-shirt, shoes, pants, etc.
# MAGIC     - CIFAR-10: 32x32x3 color, labels: automobile, frog, horse, cat, dog...
# MAGIC     - Both of them within tensorflow: `tf.keras.datasets.fashion_mnist.load_data()` & `tf.keras.datasets.cifar10.load_data()`
# MAGIC - Step 2: Build model
# MAGIC     - CNNs in this case
# MAGIC     - Funcional API: better way of creating models, Keras convention
# MAGIC - Step 3: Train the model
# MAGIC - Step 4: Evaluate the model
# MAGIC - Step 5: Make predictions
# MAGIC - Bonus: Data augmentation

# COMMAND ----------

!/databricks/python3/bin/python -m pip install --upgrade pip

# COMMAND ----------

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# more convinient:
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout # regular CNN
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D, BatchNormalization # advanced CNN
from tensorflow.keras.models import Model

# COMMAND ----------

tf.config.list_physical_devices('GPU')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fashion MNIST

# COMMAND ----------

# downlaod data
data = tf.keras.datasets.fashion_mnist.load_data()

# COMMAND ----------

# extract data
(X_train, y_train), (X_test, y_test) = data

# COMMAND ----------

# convert range from (0, 255) to (0, 1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# COMMAND ----------

print(X_train.shape)

# COMMAND ----------

# increase another dimension, as this is what the CNN expects
X_train = np.expand_dims(X_train,-1)
X_test = np.expand_dims(X_test,-1)
print(X_train.shape)

# COMMAND ----------

# store number of possible classes
K = len(set(y_train))
print('Number of possible classes', K)

# COMMAND ----------

# MAGIC %md
# MAGIC ### CNN Model

# COMMAND ----------

# build model
i = Input(shape=X_train[0].shape)
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

# COMMAND ----------

# compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# COMMAND ----------

# fit model
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15)

# COMMAND ----------

# plot loss per epoch
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.title('Loss per epoch')
plt.legend()

# COMMAND ----------

# plot accuracy per epoch
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy per epoch')
plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze Model

# COMMAND ----------

model.evaluate(X_test, y_test)

# COMMAND ----------

# Confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

# COMMAND ----------

# create a function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i,j] > thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# COMMAND ----------

# Calculate confusion matrix
y_hat = model.predict(X_test).argmax(axis=1)
cm = confusion_matrix(y_test, y_hat)

# COMMAND ----------

# classes label
classes = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

# COMMAND ----------

plot_confusion_matrix(cm, classes)

# COMMAND ----------

# show misclassified example
misclassified_idx = np.where(y_hat != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(X_test[i], cmap='gray')
plt.title(f'True label: {classes[y_test[i]]}, Predicted: {classes[y_hat[i]]}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## CIFAR-10 with advanced arquitecture

# COMMAND ----------

# downlaod data
data = tf.keras.datasets.cifar10.load_data()

# COMMAND ----------

# extract data
(X_train, y_train), (X_test, y_test) = data

# COMMAND ----------

# convert range from (0, 255) to (0, 1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# COMMAND ----------

# flatten target variable to a 1D array
y_train = y_train.flatten()
y_test = y_test.flatten()

# COMMAND ----------

print(X_train.shape)

# COMMAND ----------

# store number of possible classes
K = len(set(y_train))
print('Number of possible classes', K)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# COMMAND ----------

# MAGIC %md
# MAGIC ### CNN Model

# COMMAND ----------

# build model, inspiration from the VGG network
i = Input(shape=X_train[0].shape)
# x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
# x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
# x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)  # Not very useful, we lose information

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)  # Not very useful, we lose information

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)  # Not very useful, we lose information

# x = GlobalMaxPooling2D()(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

# COMMAND ----------

# Print a summary of the model
model.summary()

# COMMAND ----------

# compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# COMMAND ----------

# fit model
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)

# COMMAND ----------

# plot loss per epoch
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.title('Loss per epoch')
plt.legend()

# COMMAND ----------

# plot accuracy per epoch
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy per epoch')
plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze Model

# COMMAND ----------

model.evaluate(X_test, y_test)

# COMMAND ----------

# Calculate confusion matrix
y_hat = model.predict(X_test).argmax(axis=1)
cm = confusion_matrix(y_test, y_hat)

# COMMAND ----------

plot_confusion_matrix(cm, classes)

# COMMAND ----------

# show misclassified example
misclassified_idx = np.where(y_hat != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(X_test[i])
plt.title(f'True label: {classes[y_test[i]]}, Predicted: {classes[y_hat[i]]}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## CIFAR-10 with advanced arquitecture + Data Augmentation

# COMMAND ----------

# predefine again the arquitecture to start training process from scratch
i = Input(shape=X_train[0].shape)
# x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
# x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
# x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)  # Not very useful, we lose information

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)  # Not very useful, we lose information

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)  # Not very useful, we lose information

# x = GlobalMaxPooling2D()(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

# COMMAND ----------

# compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data augmentation

# COMMAND ----------

batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_generator = data_generator.flow(X_train, y_train, batch_size)
steps_per_epoch = X_train.shape[0] // batch_size
r = model.fit(train_generator, validation_data=(X_test, y_test), steps_per_epoch=steps_per_epoch, epochs=50)

# COMMAND ----------

# plot loss and accuracy per epoch
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.title('Loss per epoch')
plt.legend()

# plot accuracy per epoch
plt.subplot(1, 2, 2)
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy per epoch')
plt.legend()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze the model

# COMMAND ----------

model.evaluate(X_test, y_test)

# COMMAND ----------

# Calculate and plot confusion matrix
y_hat = model.predict(X_test).argmax(axis=1)
cm = confusion_matrix(y_test, y_hat)
plot_confusion_matrix(cm, classes)

# COMMAND ----------

# show misclassified example
misclassified_idx = np.where(y_hat != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(X_test[i])
plt.title(f'True label: {classes[y_test[i]]}, Predicted: {classes[y_hat[i]]}')

# COMMAND ----------


