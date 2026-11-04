import tensorflow as tf

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")


print(train.shape)

print(test.shape)



X_train = train.drop("label", axis=1)

y_train = train["label"]

X_test = test



X_train = X_train / 255.0

X_test = X_test / 255.0



model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(3072,)),

    tf.keras.layers.Dense(256, activation="relu"),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dense(10, activation="softmax")

])



model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)



model.fit(X_train, y_train, epochs=5, batch_size=32)



Predict using test data 

prediction = model.predict(X_test)

predicted = np.argmax(prediction[0])

print("Predicted Class:", predicted)





Display one test image 

image = X_test.iloc[0].values.reshape(3, 32, 32).transpose(1, 2, 0)

plt.imshow(image)

plt.title("Predicted Class: " + str(predicted))

plt.axis("off")

plt.show()

