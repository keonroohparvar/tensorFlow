# This is a hidden Markov Model to predict future weather

# Imports
import tensorflow_probability as tfp
import tensorflow as tf


# Loading tensorflow_probability.distribution as tfd
tfd = tfp.distributions

# Referring to the fact that the first day has an 80% chance of being cold (Encoded 0)
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])

# These is the Transition Distribution ; First [0.7, 0.3] is for cold days (encoded 0)
# and [0.8, 0.2] is for hot days (Encoded 1)
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], [0.8, 0.2]])

# This is the Observaion Distribution
observation_distribution = tfd.Normal(loc=[0.,15.], scale=[5., 10.]) # Refers to means / standard deviations

# Creating Hidden Markov Model
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps = 7 # num_steps is the number of days you want to predict
) 

mean = model.mean()

# Due to tensorflow working on a lower level, we need to evaluate in a session

with tf.compat.v1.Session() as sess:
    print(mean.numpy())

# This prints [2.9999998 4.2       4.0799994 4.0919995 4.0907993 4.090919  4.090907 ]
# these represents the predicted temperatures for the 7 days
