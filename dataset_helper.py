import os
import numpy as np
import tensorflow as tf
from PIL import Image

def init_records(dataset_path, train_size=200, test_size=20):
    """ Use init_record to create train tfrecords and test tfrecords
    """
    init_record(dataset_path, 'train.tfrecords', train_size)
    init_record(dataset_path, 'test.tfrecords', test_size)

def init_record(dataset_path, record_name, size):
    """ load images from data and create a new tfrecords
    """
    filepaths = []
    labels = []

    categories = os.listdir(dataset_path)
    for category_index, category in enumerate(categories):
        category_path = dataset_path + "/" + category + "/"
        filepaths += list("%s%d.jpg" % (category_path, i) for i in range(size))
        labels += [category_index for _ in range(size)]

    to_TFRecords(filepaths, labels, record_name)

def to_TFRecords(filepaths, labels, tfrecords_name):
    """ Create a tfrecords from given paths to images and matching labels
    """

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    writer = tf.python_io.TFRecordWriter(tfrecords_name)

    for i in range(len(labels)):

        img = Image.open(filepaths[i])
        img = np.array(img.resize((32,32)))
        img_raw = img.tostring()

        feature = {
            'label': _int64_feature(labels[i]),
            'image': _bytes_feature(img_raw)
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

    writer.close()
