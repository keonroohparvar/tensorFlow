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
def input_fn(features, labels, training=True, batch_size=256):
    # Converts inputs to a data.Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

# Because feature columns are all numerical, we can just loop through
feature_columns = []
for key in train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

# Creating the Estimator: going to create a Deep Neural Network
# The DNN has 2 hidden layers with 30 nodes and 10 hidden nodes each
classifier = tf.estimator.DNNClassifier(
    # Init Feature Columns
    feature_columns=feature_columns,
    # These are the hidden units
    hidden_units=[30, 10],
    # The classifier has to choose between 3 classes
    n_classes=3
)

# Training the Model
classifier.train(
    input_fn = lambda:input_fn(train, train_y, training=True),
    steps = 500
)

# Testing the model against known values
eval_result = classifier.evaluate(input_fn = lambda:input_fn(test, test_y, training=False))

print('\nTest accuracy: {accuracy:0.3f}'.format(**eval_result))