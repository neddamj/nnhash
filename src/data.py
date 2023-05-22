import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from utils import save_img, resize_imgs

class CIFAR10:
    def load(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        return x_train

    def display(self, data, label):
        dist = np.random.randint(low=0, high=10000, size=100)
        plt.figure(figsize=(10, 4))
        for i in range(10):
            plt.subplot(2, 5, 1 + i)
            image = data[dist[i]]
            lbl = label[dist[i]]
            plt.imshow(image)
        plt.tight_layout()
        plt.show()

    def save_to_disk(self, data, path='../images', num_images=100):
        assert data.shape[-1] == 3, "Please pass the image data to the function rather than the labels"
        dist = np.random.randint(low=0, high=10000, size=num_images)
        for i in range(num_images):
            img = data[dist[i]]
            save_img(f'{path}{i+1}.jpeg', img)
        print('[INFO] Images saved')

    

class IMAGENETTE:
    def load(self, split='train'):
        ds = tfds.load(
            name='imagenette/320px-v2',
            split=split,
            shuffle_files=True
        )
        return ds

    def save_to_disk(self, data, path='../images', num_images=100):
        assert isinstance(data, tf.data.Dataset)
        dist = np.random.randint(low=0, high=10000, size=num_images)
        for i, example in enumerate(data):
            img = example['image']
            save_img(f'{path}{i+1}.jpeg', img)
            resize_imgs(f'{path}{i+1}.jpeg', new_size=(224,224))
            if i == num_images-1:
                break
        print('[INFO] Images saved')

if __name__ == "__main__":
    data = IMAGENETTE()
    ds = data.load()
    #data.display(ds, x_test)

    folder_path = '../images/'
    data.save_to_disk(ds, folder_path)