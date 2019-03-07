import tensorflow as tf

class Model:
    def __init__(self, batch_size):
        self.__batch_size = batch_size

    def get_model(self):
        initial_model = tf.keras.applications.NASNetLarge(
            input_shape=(331, 331, 3),
            include_top=False,
        )

        for l in initial_model.layers:
            l.trainable = True

        input = tf.keras.layers.Input(shape=(400, 400, 3), batch_size=self.__batch_size)
        x = tf.keras.layers.Lambda(lambda img: tf.image.resize_bicubic(img, size=(331, 331)))(input)
        x = initial_model(x)
        x = tf.keras.layers.Flatten(data_format='channels_last')(x)
        x = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(x)

        model = tf.keras.models.Model(inputs=input, outputs=x)
        model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.5, nesterov=True), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
        return model
