import tensorflow as tf
import numpy as np
import math
from google.cloud import storage

basepath = "gs://ericdhiggins"

class Input:
    def __init__(self):
        client = storage.Client()
        self.__bucket = client.get_bucket('ericdhiggins')

    def load_data(self, index, path):
        data = self.__bucket.get_blob(path).download_as_string()
        data = data.decode().split('\n')
        data = list(map(lambda x: x.split(','), data))[1:]
        data = list(filter(lambda x: len(x) > (5 + index) and x[3] == 'Frontal' and x[5+index] in ['1.0', '0.0'], data))
        image_names = list(map(lambda x: [x[0]], data))

        labels = self.get_labels(data, index)
        labels = np.reshape(np.asarray(labels), (-1, 1))

        self.step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
        return tf.data.Dataset.from_tensor_slices((image_names, labels))

    def get_labels(self, labels, index):
        return list(map(lambda p: math.floor(float(p[5+index])) if len(p[5+index].strip()) > 0 else -1, labels))

    def process_image(self, features, label):
        file = tf.read_file(tf.strings.join([basepath, features[0]], separator='/'))
        image = tf.image.decode_jpeg(file, channels=3)
        image = tf.image.resize_image_with_crop_or_pad(image, 400, 400)
        return image, label

    def get_dataset(self, path, batch_size, col_index):
        dataset = self.load_data(col_index, path)
        dataset = dataset.map(self.process_image)
        dataset = dataset.shuffle(buffer_size=batch_size*2)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size*2)
        return dataset

    def train_input_fn(self, batch_size, col_index):
        def input():
            return self.get_dataset('data/train.csv', batch_size, col_index)
        return input
    
    def dev_input_fn(self, batch_size, col_index):
        def input():
            return self.get_dataset('data/valid.csv', batch_size, col_index)
        return input
