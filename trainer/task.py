import os
from .model import Model
from .input import Input

checkpoint_path = 'models/inceptionv3'
job_dir = 'gs://ericdhiggins/models/inceptionv3'

batch_size = 56
col_index = 1

def train_and_evaluate():
    print('Building model...')

    estimator = Model().get_model()
    input = Input()

    print('Training...')
    estimator.train(
        input_fn=input.train_input_fn(batch_size, col_index),
        steps=100
    )

    estimator.evaluate(input_fn=input.dev_input_fn(batch_size, col_index), steps=4)

def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(
            os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())

train_and_evaluate()