import os
import numpy as np
import tensorflow as tf
from PIL import Image

def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image'], tf.uint8)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label

def read_tfrecord(filename):
    dataset = tf.data.TFRecordDataset(filename)

    dataset = dataset.map(decode)

    dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(2)
    dataset = dataset.batch(32)

    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

def init_records(dataset_path, train_size=200, test_size=20):
    """ Use init_record to create train tfrecords and test tfrecords
    """
    init_record(dataset_path, 'train.tfrecord', train_size)
    init_record(dataset_path, 'test.tfrecord', test_size)

def init_record(dataset_path, record_name, size):
    """ load images from data and create a new tfrecord
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
    """ Create a tfrecord from given paths to images and matching labels
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
