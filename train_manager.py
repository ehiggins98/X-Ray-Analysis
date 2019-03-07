import tensorflow as tf
from input import Input
from model import Model
import os
import numpy as np

def train():
    batch_size = 56
    col_index = 1
    
    def input_generator():
      input = Input().train_input_fn(batch_size, col_index)()
      iter = input.make_initializable_iterator()
      while True:
        batch_x, batch_y = iter.get_next()
        batch_x.set_shape((batch_size, 400, 400, 3))
        batch_y.set_shape((batch_size, 1))
        yield batch_x, batch_y
    
    print('Building model...')
    
    tpu_model = tf.contrib.tpu.keras_to_tpu_model(
      Model().get_model(),
      strategy=tf.contrib.tpu.TPUDistributionStrategy(
          tf.contrib.cluster_resolver.TPUClusterResolver(tpu='ericdhiggins', zone='us-central1-b')))
    
    print('Training...')
    tpu_model.fit_generator(
      input_generator(),
      steps_per_epoch=10,
    )
if __name__ == '__main__':
    train()


if __name__ == '__main__':
    train()
