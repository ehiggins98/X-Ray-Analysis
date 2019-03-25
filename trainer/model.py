import tensorflow as tf
from keras import backend as K

class Model:
    def get_model(self):
        initial_model = tf.keras.applications.DenseNet121(
            input_shape=(400, 400, 3),
            include_top=False,
        )

        for l in initial_model.layers:
            l.trainable = True

        input = tf.keras.layers.Input(shape=(400, 400, 3))
        x = initial_model(input)
        x = tf.keras.layers.Flatten(data_format='channels_last')(x)
        x = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(x)

        model = tf.keras.models.Model(inputs=input, outputs=x)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])

        return tf.keras.estimator.model_to_estimator(keras_model=model, model_dir='gs://ericdhiggins/job17')