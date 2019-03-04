import tensorflow as tf
from input import Input
from model import Model
import numpy as np

def train():
    estimator = Model().get_model()
    input = Input()

    for i in range(3):
        estimator.train(
            input_fn=input.train_input_fn(batch_size=2, col_index=1),
            steps=50
        )

        accuracy = estimator.evaluate(input_fn=input.dev_input_fn(batch_size=5, col_index=1), steps=1)
        print(accuracy)
    
if __name__ == '__main__':
    train()