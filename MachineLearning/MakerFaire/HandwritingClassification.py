from extra_keras_datasets import emnist
from matplotlib import pyplot as plt

#load emnist dataset
(x_train, y_train), (x_test, y_test) = emnist.load_data(type='balanced')

#print size of each set and image size
print('Training set: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test set: X=%s, y=%s' % (x_test.shape, y_test.shape))

#test images
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i], cmap = plt.get_cmap('gray'))
plt.show()