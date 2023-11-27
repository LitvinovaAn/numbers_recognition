import cv2
import numpy as np
import tensorflow as tf

from augment import augmentation
from loader import load


class DataGenerator(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""
    def __init__(self, batch_size, images, annotation, input_size=(64, 64), shuffle=False):
        self.batch_size = batch_size
        self.images = images
        self.annotation = annotation
        self.shuffle = shuffle
        self.input_size = input_size
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return len(self.images) // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        i = idx * self.batch_size
        indexes = self.indexes[i: i + self.batch_size]
        batch_input_img_paths = [self.images[j] for j in indexes]
        batch_annotation = [self.annotation[j] for j in indexes]
        x = []
        y = []

        for j, path in enumerate(batch_input_img_paths):
            img = cv2.imread(path)
            # img = augmentation(img)
            img = cv2.resize(img, self.input_size)
            x.append(img)
            y.append(batch_annotation[j])

        x = np.array(x, dtype="float32")
        y = np.array(y, dtype="float32")

        return x, y


if __name__ == '__main__':
    path = 'numbers101'
    train_image, train_label, valid_image, valid_label = load(path)

    train_generator = DataGenerator(8, train_image, train_label, input_size=(64, 64), shuffle=True)
    valid_generator = DataGenerator(8, valid_image, valid_label, input_size=(64, 64), shuffle=False)

    for i in range(train_generator.__len__()):
        x, y = train_generator.__getitem__(i)

        for xi, yi in zip(x, y):
            img = xi.astype("uint8")
            cv2.putText(img, f"{yi}", (10, 25), 1, 1.4, (255, 0, 0), 3)
            cv2.putText(img, f"{yi}", (10, 25), 1, 1.4, (255, 255, 0), 1)
            cv2.imshow("img", img)
            cv2.waitKey(0)



