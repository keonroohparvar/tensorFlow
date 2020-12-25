# This is a neural network from tensorflow and Keras. This will guess fashion clothes using the 
# MNIST Fashion dataset.

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Loading the dataset (note it is already loaded into Keras)
fashion_mnist = keras.datasets.fashion_mnist

# Partition the testing and training data into apropriate tuples
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# This prints (60000, 28, 28) which means we have 60,000 images that are all 28x28 pixels
print(train_images.shape)

# This references the 0th images, and the 23x23 pixel
# This is a value between 0 and 255, which 0 is black and 255 is white so this is greyscale
print(train_images[0, 23, 23])

# This prints the first 10 labels
# These are representations of all 10 classes, each has a number attached to it 0-9
print(train_labels[:10])

# Array of the classnames
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
"Sneaker", "Bag", "Ankle boot"]

# This is matplotlib used to display our image
"""
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
"""

# The following preprocesses the data to between 0 and 1 for the NN to handle it better
train_images = train_images / 255
test_images = test_images / 255

# Creating the model
model = keras.Sequential([
    # This is our input layer. Flatten allows us to take a shape 28 by 28 and put it into a 28^2 object
    keras.layers.Flatten(input_shape=(28, 28)),

    # This is our first hidden layer (layer 2) and it is dense. This meanse
    # all the neurons in the previous layer are connected to every neuron in this layer.
    # The 128 is random, but you need to decide what the number of layers should be based off
    # of the problem.
    keras.layers.Dense(128, activation='relu'),

    # This is our output layer, note that it is dense as well. There are 10 neurons because there are
    # 10 classes. Softmax makes it so that all of the probabilities add up to 1
    keras.layers.Dense(10, activation='softmax')
])

# This is compiling the model with the optimizer, loss function, and metrics
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Training the model
model.fit(train_images, train_labels, epochs=1)

# This is testing the accuracy of the model. Verbose is whether or not we are looking at the information
# in the console.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

# Going to predict on all of the test images
# predictions is an array of the probability of each class; the highest number is the one that the model thinks the object is
predictions = model.predict(test_images)

# The argmax method returns the index of the largest number; that means for prediction 0, the largest class 
# probability is np.argmax(predictions[0])
print(np.argmax(predictions[0]))