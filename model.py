import tensorflow as tf

class Model:
    def get_model(self):
        initial_model = tf.keras.applications.NASNetLarge(
            input_shape=(331, 331, 3),
            include_top=False,
        )

        for l in initial_model.layers:
            l.trainable = True

        input = tf.keras.layers.Input(shape=(400, 400, 3))
        x = tf.keras.layers.Lambda(lambda img: tf.image.resize_bicubic(img, size=(331, 331)))(input)
        x = initial_model(x)
        x = tf.keras.layers.Flatten(data_format='channels_last')(x)
        x = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(x)

        model = tf.keras.models.Model(inputs=input, outputs=x)
        model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])

        return tf.keras.estimator.model_to_estimator(keras_model=model, model_dir='models/nasnet')