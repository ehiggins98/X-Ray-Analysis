import tensorflow as tf

class Model:
    def get_model(self):
        initial_model = tf.keras.applications.InceptionV3(
            input_shape=(400, 400, 3),
            include_top=False,
        )

        for l in initial_model.layers:
            l.trainable = True

        x = tf.keras.layers.Flatten(data_format='channels_last')(initial_model.layers[-1].output)
        x = tf.keras.layers.Dense(units=1)(x)

        model = tf.keras.models.Model(inputs=initial_model.input, outputs=x)
        model.compile(optimizer=tf.train.MomentumOptimizer(0.01, 0.5, use_nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])
        return model
