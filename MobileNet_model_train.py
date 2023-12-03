from matplotlib import pyplot as plt
from generator import DataGenerator
from loader import load
import tensorflow as tf


def train_numbers(
        data_path,
        input_size=(64, 64),
        weight_path="numbers.h5",
        batch_size=64,
        learning_rate=0.0001,
        epochs=200,
        load_weights=None
):

    train_images, train_labels, valid_images, valid_labels = load(data_path)

    train_generator = DataGenerator(batch_size, train_images, train_labels, shuffle=True)
    valid_generator = DataGenerator(batch_size, valid_images, valid_labels, shuffle=False)

    base_model = tf.keras.applications.MobileNetV2(input_shape=input_size + (3,),
                                                   include_top=False,
                                                   weights='imagenet'
                                                   )
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(101, activation="softmax")
    ])

    if load_weights is not None:
        model.load_weights(load_weights)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(weight_path, save_best_only=True, period=1, save_weights_only=True)
    ]

    history = model.fit(train_generator, validation_data=valid_generator, epochs=epochs, callbacks=callbacks)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    train_numbers("numbers101", (64, 64))


