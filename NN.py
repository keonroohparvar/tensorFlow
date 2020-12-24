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

print(train_images.shape)


