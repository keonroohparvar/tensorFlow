import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc


# These two methods create dataframes.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

y_train = dftrain.pop('survived') # This removes dataframe of survived and puts it in y_train 
y_eval = dfeval.pop('survived') # This removes dataframe of survived and puts it in y_eval

# To find one row, you use dftrain.loc(index) to point to that specific index
print(dftrain.loc[0])

# to reference an entire column, you would use dftrain[columnName]
print(dftrain["sex"], "\n")
# Can call dftrain.describe() to print some statistics


# These are represented by category
CATEGORICAL_COLUMNS = ["n_siblings_spouses", "sex", "parch", "class", "deck", "embark_town", "alone"] 
# These are represented by numerical values
NUMERICAL_COLUMNS = ["age", "fare"] 

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() # This method gets a list of all unique values
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
for feature_name in NUMERICAL_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# The following is an example of an input function
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) # create a tf.data.Dataset object with data and its labels
        if shuffle:
            ds = ds.shuffle(1000) # Randomize order of data
        ds = ds.batch(batch_size).repeat(num_epochs) # Split data into batches of 32 and repeat number of epoch times
        return ds # Return a batch of the dataset
    return input_function

train_input_fn = make_input_fn(dftrain, y_train) # makes input for training data
eval_input_fn = make_input_fn(dfeval, y_eval) # makes input for evaluation data

# Creating linear model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# Training the model
linear_est.train(train_input_fn)

# Get model metrics / stats by testing on test data
result = linear_est.evaluate(eval_input_fn)

# The following will print the result of the accuracy
#clear_output() # Clears console output
#print(result)

# The following is how you predict in Linear Models
result = list(linear_est.predict(eval_input_fn))
print(result[0])