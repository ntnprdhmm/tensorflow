import cv2
import numpy as np
import os
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tqdm import tqdm
import math

MAX_NB_IMAGES = 2000
DATASET_DIR = 'dataset/PetImages'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'dogs-vs-cats-convnet'

train_data = []
test_data = []

if not (os.path.exists('train_data.npy') and os.path.exists('test_data.npy')):
    def create_label(category):
        """ Create an one-hot encoded vector from image name """
        if category == 'Cat':
            return np.array([1,0])
        elif category == 'Dog':
            return np.array([0,1])
        return np.array([0, 0])

    def create_data(category, start, n):
        testing_data = []
        label = create_label(category)
    	# read images after the training images
        for i in tqdm(range(start, start+n)):
            path = os.path.join(DATASET_DIR + '/' + category, str(i) + '.jpg')
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img_data is not None:
                img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
                testing_data.append([np.array(img_data), label])
        shuffle(testing_data)
        return testing_data

    def count_images(category):
    	""" count the number of images of this category """
    	return len(os.listdir(DATASET_DIR + '/' + category))

    # count the number of images available
    nb_cats = count_images('Cat')
    nb_dogs = count_images('Dog')

    # don't take too many images (else, it will be too slow)
    nb_cats = min(nb_cats, MAX_NB_IMAGES)
    nb_dogs = min(nb_dogs, MAX_NB_IMAGES)

    # split to train/test
    nb_cats_train = math.floor(nb_cats * 0.90)
    nb_cats_test = math.floor(nb_cats * 0.10)
    nb_dogs_train = math.floor(nb_dogs * 0.90)
    nb_dogs_test = math.floor(nb_dogs * 0.10)

    # create training datasets
    cats_train_data = create_data('Cat', 1, nb_cats_train)
    dogs_train_data = create_data('Dog', 1, nb_dogs_train)
    train_data = cats_train_data + dogs_train_data
    shuffle(train_data)

    # create test datasets
    cats_test_data = create_data('Cat', nb_cats_train+1, nb_cats_test)
    dogs_test_data = create_data('Dog', nb_dogs_train+1, nb_dogs_test)
    test_data = cats_test_data + dogs_test_data
    shuffle(test_data)

    np.save('train_data.npy', train_data)
    np.save('test_data.npy', test_data)
else:
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')

X_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train_data]

X_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test_data]

tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
    validation_set=({'input': X_test}, {'targets': y_test}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
