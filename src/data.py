import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import save_img

class CIFAR10:
    def __init__(self):
        pass

    def load(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        return (x_train, y_train), (x_test, y_test)

    def save_to_disk(self, data, path='../images', num_images=100):
        assert data.shape[-1] == 3, "Please pass the image data to the function rather than the labels"
        dist = np.random.randint(low=0, high=10000, size=num_images)
        for i in range(num_images):
            img = data[dist[i]]
            save_img(f'{path}{i+1}.jpeg', img)
        print('[INFO] Images saved')

    def display(self, x_train, y_train, x_test, y_test):
        dist = np.random.randint(low=0, high=10000, size=100)
        plt.figure(figsize=(10, 4))
        for i in range(10):
            plt.subplot(2, 5, 1 + i)
            image = x_test[dist[i]]
            label = y_test[dist[i]]
            plt.imshow(image)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    cifar10 = CIFAR10()
    (x_train, y_train), (x_test, y_test) = cifar10.load()
    print(y_train.shape, y_test.shape)
    cifar10.display(x_train, y_train, x_test, y_test)

    folder_path = '../images/'
    cifar10.save_to_disk(x_train, folder_path)