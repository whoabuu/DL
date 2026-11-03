
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import string

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from numpy.linalg import norm

text =""" Deep learning is a specialized, more advanced type of machine learning that mimics how the human brain processes information. It uses multi-layered structure called artificial neural networks to automatically find complex patterns in data, such as recognizing a specific face in a crowd or understanding spoken language. While regular machine learning often needs human guidance to label key features, deep learning can automatically extract features directly from raw data like images, audio, and text, making it the driving force behind modern breakthroughs like generative AI, self-driving cars, and digital assistants. """

words = text.lower().split()

print("Total Words: ",len(words))

words=[word.strip(string.punctuation)
      for word in words]

words = [word for word in words if word]

print("Clean words: ")
print(words)

vocab = sorted(set(words))
vocab_size = len(vocab)

word_to_index = {word: i for i, word in enumerate(vocab)}

index_to_word = {i:word for i , word in enumerate(vocab)}

print("Vocab Size : ",vocab_size)
print(vocab)

window_size = 2

x = []
y = []

for i in range(window_size, len(words) - window_size):
    context_words = (
        words[i-window_size:i] +
        words[i+1:i+window_size+1]
    )

    target_word = words[i]

    context_vector = np.zeros(vocab_size)

    for word in context_words:
        context_vector[word_to_index[word]] += 1

    context_vector = context_vector / len(context_words)

    target_index = word_to_index[target_word]

    x.append(context_vector)
    y.append(target_index)


x = np.array(x)

y = to_categorical(y, num_classes=vocab_size)

print("x shape:", x.shape)
print("y shape:", y.shape)

model = Sequential([
    Dense(50,input_shape=(vocab_size,),activation='relu'),
    Dense(vocab_size,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history=model.fit(x,y,epochs=60,batch_size=16,verbose=1)

def predict_word(context_words):

    context_vector = np.zeros(vocab_size)

    for word in context_words:
        if word in word_to_index:
            context_vector[word_to_index[word]] += 1

    context_vector = context_vector / len(context_words)

    prediction = model.predict(
        np.array([context_vector]),
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    return index_to_word[predicted_index]

context = ["deep", "is", "type", "machine"]


predicted_word = predict_word(context)

print("Context words:", context)
print("Predicted word:", predicted_word)

