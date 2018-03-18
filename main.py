from dataset_helper import init_records, read_tfrecord
import tensorflow as tf

if __name__ == '__main__':
    #init_records("./dataset/PetImages")
    image, label = read_tfrecord('train.tfrecord')
