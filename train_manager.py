import tensorflow as tf
from input import Input
from model import Model
import numpy as np

def train():
    batch_size = 50
    model = Model().get_model()
    estimator = tf.contrib.tpu.keras_to_tpu_model(
        model=model,
        strategy=tf.contrib.tpu.TPUDistributionStrategy(using_single_core=True)
    )

    estimator.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.5, nesterov=True), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])

    def train_gen(batch_size):
        dataset = Input().train_input_fn(batch_size, 0)()
        iter = dataset.make_initializable_iterator()

        while True:
            yield iter.get_next()

    x_val, y_val = Input().dev_input_fn(234, 0)().make_initializable_iterator().get_next()

    estimator.fit_generator(
        train_gen(batch_size),
        epochs=2,
        steps_per_epoch=4470,
        validation_data={x_val, y_val}
    )
    
if __name__ == '__main__':
    train()