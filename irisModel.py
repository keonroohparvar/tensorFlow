# Imports
import tensorflow as tf
import pandas as pd

# These are the headers in our dataframe
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
 
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# The following are constants that will save time
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

# We will use Keras (a tensorflow model) to help read data into a pandas dataframe
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# Seperating the thing we eventually want our model to predict
train_y = train.pop('Species')
test_y = test.pop('Species')

# Input Function