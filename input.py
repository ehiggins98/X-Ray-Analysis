import tensorflow as tf
import numpy as np
import math

class Input:
    def load_data(self, index, path):
        data = open(path).readlines()
        data = list(map(lambda x: x.split(','), data))[1:]
        data = list(filter(lambda x: x[3] == 'Frontal', data))
        image_names = list(map(lambda x: [x[0]], data))

        labels = self.get_labels(data, index)
        labels = np.reshape(np.asarray(labels), (-1, 1))

        self.step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
        return tf.data.Dataset.from_tensor_slices((image_names, labels))

    def get_labels(self, labels, index):
        return list(map(lambda p: math.floor(float(p[5+index])) if len(p[5+index].strip()) > 0 else -1, labels))

    def process_image(self, features, label):
        image = tf.image.decode_jpeg(tf.read_file('/home/ericdhiggins/' + features[0]), channels=3)
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
        def input(params):
            return self.get_dataset('/home/eric/CheXpert-v1.0-small/train.csv', batch_size, col_index)
        return input
    
    def dev_input_fn(self, batch_size, col_index):
        def input(params):
            return self.get_dataset('/home/eric/CheXpert-v1.0-small/valid.csv', batch_size, col_index)
        return input
