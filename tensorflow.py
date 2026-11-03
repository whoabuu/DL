import tensorflow as tf
import numpy as np

# Create tensor
x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

print("Tensor:", x)
print("Shape:", x.shape)
print("Data Type:", x.dtype)

# Basic operations
print("Addition:", x + x)
print("Subtraction:", x - x)
print("Multiplication:", 5 * x)

# Transpose and matrix multiplication
print("Transpose:", tf.transpose(x))
print("Matrix Multiplication:", x @ tf.transpose(x))

# Concatenation
print("Concatenation:", tf.concat([x, x], axis=0))

# Softmax
print("Softmax:", tf.nn.softmax(x))

# Variables
W = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.zeros([1]))

print("Weight:", W)
print("Bias:", b)




import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv('diabetes.csv')

# Separate input and output
x = df.iloc[:, 0:8]
y = df.iloc[:, 8]

# Create model
model = Sequential()

# Add layers
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.fit(x, y, epochs=10, batch_size=10)

# Evaluate model
loss, accuracy = model.evaluate(x, y)

print("Accuracy:", accuracy)
