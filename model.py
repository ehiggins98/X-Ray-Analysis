import tensorflow as tf

class Model:
    def __init__(self, batch_size):
        self.__batch_size = batch_size

    def get_model(self, features, labels, mode, params):
        labels = tf.cast(labels, tf.float32)
        initial_model = tf.keras.applications.NASNetLarge(
            input_shape=(331, 331, 3),
            include_top=False,
        )

        for l in initial_model.layers:
            l.trainable = True

        input = tf.keras.layers.Input(shape=(400, 400, 3), tensor=features, batch_size=self.__batch_size)
        x = tf.keras.layers.Lambda(lambda img: tf.image.resize_bicubic(img, size=(331, 331)))(input)
        x = initial_model(x)
        x = tf.keras.layers.Flatten(data_format='channels_last')(x)
        x = tf.keras.layers.Dense(units=1)(x)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=labels)

        optimizer = tf.train.MomentumOptimizer(0.01, 0.5, use_nesterov=True)

        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            predictions={
                "classes": tf.math.round(x),
                "probabilities": tf.math.sigmoid(x)
            },
            eval_metrics=(
                tf.metrics.accuracy,
                [labels, x, None, None, None, None]
            )
        )
