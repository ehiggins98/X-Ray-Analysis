import tensorflow as tf
from input import Input
from model import Model
import numpy as np

def train():
    estimator = Model().get_model()
    input = Input()

    for i in range(20):
        estimator.train(
            input_fn=input.train_input_fn(batch_size=10, col_index=1),
            steps=100
        )

        accuracy = estimator.evaluate(input_fn=input.dev_input_fn(batch_size=10, col_index=1), steps=24)
        print(accuracy)
    
if __name__ == '__main__':
    train()