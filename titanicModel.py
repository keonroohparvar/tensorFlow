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


