import tensorflow as tf

class Model:
    def __init__(self):
        self.accuracy = None

    def get_model(self):
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Conv2D(
                filters=64,
                input_shape=(400, 400, 1),
                kernel_size=5,
                padding="same",
                data_format="channels_last",
                activation=tf.nn.relu,
                use_bias=True
            )
        )

        model.add(
            tf.keras.layers.Reshape(
                (400*400*64,)
            )
        )

        model.add(
            tf.keras.layers.Dense(
                units=1,
                activation=tf.math.sigmoid,
                use_bias=True
            )
        )

        model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metric='BinaryAccuracy')

        return tf.keras.estimator.model_to_estimator(keras_model=model, model_dir='models/model1')