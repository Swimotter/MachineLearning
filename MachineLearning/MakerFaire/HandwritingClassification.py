from extra_keras_datasets import emnist
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import keras as ks

def load_data():
    #load emnist dataset
    (x_train, y_train), (x_test, y_test) = emnist.load_data(type='byclass')
    #print size of each set and image size
    print('Training set: X=%s, y=%s' % (x_train.shape, y_train.shape))
    print('Test set: X=%s, y=%s' % (x_test.shape, y_test.shape))
    #resize data to be a single type - decrease run time
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    #hot encode targets
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    return x_train, y_train, x_test, y_test



print(load_data())

#print size of each set and image size
print('Training set: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test set: X=%s, y=%s' % (x_test.shape, y_test.shape))

#test images
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i], cmap = plt.get_cmap('gray'))
plt.show()

