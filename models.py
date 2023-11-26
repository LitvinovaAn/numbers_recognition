import keras
import tensorflow as tf


def numbers(size):
    input1 = tf.keras.layers.Input(size)

    # Feature extractor
    x = tf.keras.layers.Conv2D(32, (5, 5), padding="same")(input1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.GlobalMaxPooling2D()(x)

    # Classifier
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    out = tf.keras.layers.Dense(101, activation="softmax")(x)

    model = tf.keras.Model(input1, out)

    return model


if __name__ == "__main__":
    my_model = numbers((64, 64, 3))
    my_model.summary()
