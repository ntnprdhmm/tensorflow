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

MAX_NB_IMAGES = 100
DATASET_DIR = 'dataset/PetImages'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'dogs-vs-cats-convnet'

if not (os.path.exists('train_data.npy') and os.path.exists('test_data.npy')):
    def create_label(category):
        """ Create an one-hot encoded vector from image name """
        if category == 'Cat':
            return np.array([1,0])
        elif category == 'Dog':
            return np.array([0,1])
        return np.array([0, 0])

    def create_train_data(category, n):
        training_data = []
        label = create_label(category)
    	# read images from id 1 to n
        for i in tqdm(range(n+1)):
            path = os.path.join(DATASET_DIR + '/' + category, str(i) + '.jpg')
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img_data), label])
        shuffle(training_data)
        return training_data

    def create_test_data(category, start, n):
        testing_data = []
    	# read images after the training images
        for i in tqdm(range(start, start+n)):
            path = os.path.join(DATASET_DIR + '/' + category, str(i) + '.jpg')
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            testing_data.append([np.array(img_data), path])
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
    cats_train_data = create_train_data('Cat', nb_cats_train)
    dogs_train_data = create_train_data('Dog', nb_dogs_train)
    train_data = cats_train_data + dogs_train_data
    train_data = shuffle(train_data)

    # create test datasets
    cats_test_data = create_test_data('Cat', nb_cats_train+1, nb_cats_test)
    dogs_test_data = create_test_data('Dog', nb_dogs_train+1, nb_dogs_test)
    test_data = cats_test_data + dogs_test_data
    test_data = shuffle(test_data)

    np.save('train_data.npy', train_data)
    np.save('test_data.npy', test_data)
else:
    print("ok")
