import numpy as np
from random import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

#array([[0.1, 0.2], [.2, .2]])
# array([[0.3], [0.4]])
def generate_dataset(num_samples, test_size):
  x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
  y = np.array([[i[0] + i[1]] for i in x])
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
  return x_train, x_test, y_train, y_test

if __name__=="__main__":
  x_train, x_test, y_train, y_test = generate_dataset(5000, 0.3)
  print("x_test: \n {}".format(x_test))
  print("y_test: \n {}".format(y_test))

# build model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])
optimiser = tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(optimizer=optimiser, loss="MSE")
# compile model
# train model
model.fit(x_train, y_train, epochs=100)
# evaluate model
print("\nmodel evaluation:")
model.evaluate(x_test, y_test, verbose=1)
# make predicions
data = np.array([[0.1, 0.2], [0.2, 0.2]])
predictions = model.predict(data)

print("\n Some predictions:")
for d, p in zip(data, predictions):
  print("{} + {} = {}".format(d[0], d[1], p[0]))
