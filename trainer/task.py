import os
from .model import Model
from .input import Input
import tensorflow as tf

batch_size = 16
col_index = 10
eval_after_steps = 200

def train_and_evaluate():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    print('Building model...')

    estimator = Model().get_model()
    input = Input()

    print('Training...')
    for _ in range(1):
        estimator.train(
            input_fn=input.train_input_fn(batch_size, col_index),
            steps=eval_after_steps
        )

        estimator.evaluate(input_fn=input.dev_input_fn(batch_size, col_index), steps=int(250/batch_size))

def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(
            os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())

train_and_evaluate()