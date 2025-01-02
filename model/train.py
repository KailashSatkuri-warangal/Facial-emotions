import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import cv2
import os
data = pd.read_csv('model/fer2013.csv')  
X = []
y = []
for i in range(data.shape[0]):
    pixels = data['pixels'][i].split()
    pixels = np.array(pixels, dtype='float32')
    pixels = np.reshape(pixels, (48, 48, 1))  # Reshape to 48x48 grayscale image
    X.append(pixels)
    y.append(data['emotion'][i])

X = np.array(X)
y = np.array(y)
X = X / 255.0
y = to_categorical(y, num_classes=7)  
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 classes for the 7 emotions

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=64)
model.save('model/facial_expression_model.h5')