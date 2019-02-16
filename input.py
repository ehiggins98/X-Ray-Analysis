import tensorflow as tf
import cv2 as cv
import numpy as np
import math

class Input:
    dataset_path = '/home/eric/Downloads/CheXpert-v1.0-small'

    train_list = open(f'{dataset_path}/train.csv').readlines()
    train_list = list(map(lambda x: x.split(','), train_list))

    dev_list = open(f'{dataset_path}/valid.csv').readlines()
    dev_list = list(map(lambda x: x.split(','), dev_list))

    def __init__(self):
        self.__dataset_path = '/home/eric/Downloads'
        self.__num_features = 14

        self.__train_list = open(f'{self.__dataset_path}/CheXpert-v1.0-small/train.csv').readlines()
        self.__train_list = list(map(lambda x: x.split(','), self.__train_list))[1:]
        self.__train_labels = tf.convert_to_tensor(self.get_labels(self.__train_list))
        self.__train_list = tf.convert_to_tensor(self.__train_list, dtype=tf.string)

        self.__dev_list = open(f'{self.__dataset_path}/CheXpert-v1.0-small/valid.csv').readlines()
        self.__dev_list = list(map(lambda x: x.split(','), self.__dev_list))[1:]
        self.__dev_labels = tf.convert_to_tensor(self.get_labels(self.__dev_list))

        self.step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)

    def get_labels(self, index):
        return list(map(lambda p: list(map(lambda f: math.floor(float(f)) if len(f.strip()) > 0 else -1, p[5:])), index))

    def train_input_fn(self, batch_size, sess):
        new_features = tf.Variable(initial_value=tf.zeros((batch_size, 400, 400), dtype=tf.uint8), trainable=False, dtype=tf.uint8)

        def read_image(index, start, features):
            image = tf.image.decode_jpeg(tf.read_file(self.__dataset_path + '/' + self.__train_list[index][0]), channels=3)
            image = tf.image.rgb_to_grayscale(image)
            tf.assign(new_features, features)
            image = tf.pad(image, [[400-tf.shape(image)[0], 0], [400-tf.shape(image)[1], 0], [0, 0]])
            image = tf.reshape(image, (400, 400))

            return index + 1, start, tf.assign(new_features[index-start], image)

        def input():
            start = tf.mod(tf.math.multiply(self.step, batch_size), tf.shape(self.__train_list)[0])
            end = tf.math.add(start, batch_size)

            index = tf.Variable(initial_value=start, trainable=False, dtype=tf.int32)
            features = tf.Variable(initial_value=tf.zeros((batch_size, 400, 400), dtype=tf.uint8), trainable=False, dtype=tf.uint8)

            index, start, features = tf.while_loop(lambda index, start, features: tf.math.less(index, end), read_image, [index, start, features])
            
            tf.assign(self.step, self.step+1)
            return features, self.__train_labels[start:end]
        
        return input

if __name__ == '__main__':
    with tf.Session() as sess:
        input = Input()
        features, labels = input.train_input_fn(5, sess)()
        sess.run(tf.global_variables_initializer())
        f, l = sess.run([features, labels])
        
        print(l)