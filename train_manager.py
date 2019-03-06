import tensorflow as tf
from input import Input
from model import Model
import os
import numpy as np

def train():
    batch_size = 50
    col_index = 1

    run_config = tf.contrib.tpu.RunConfig(
        model_dir='models/nasnet',
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True
        ),
        tpu_config=tf.contrib.tpu.TPUConfig()
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=Model().get_model,
        use_tpu=False,
        config=run_config,
        train_batch_size=batch_size,
        eval_on_tpu=False
    )

    input = Input()

    for _ in range(10):
        estimator.train(input_fn=input.train_input_fn(batch_size, col_index))

if __name__ == '__main__':
    train()
