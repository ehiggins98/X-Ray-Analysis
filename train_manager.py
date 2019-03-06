import tensorflow as tf
from input import Input
from model import Model
import os
import numpy as np

def train():
    batch_size = 56
    col_index = 1

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tf.contrib.cluster_resolver.TPUClusterResolver(tpu='ericdhiggins', zone='us-central1-b'),
        model_dir='models/nasnet',
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True
        ),
        tpu_config=tf.contrib.tpu.TPUConfig()
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=Model(batch_size).get_model,
        use_tpu=True,
        config=run_config,
        train_batch_size=batch_size,
        eval_on_tpu=True
    )

    input = Input()

    for _ in range(10):
        estimator.train(input_fn=input.train_input_fn(batch_size, col_index), steps=100)

if __name__ == '__main__':
    train()
