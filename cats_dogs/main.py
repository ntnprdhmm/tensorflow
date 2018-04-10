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
import re

# 12500 images available per category
MAX_NB_IMAGES_PER_CATEGORY = 800
NB_IMAGES_FOR_VALIDATION = 300
NB_IMAGES_FOR_TEST = 100

DATASET_DIR = 'dataset/PetImages'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'dogs-vs-cats-convnet'

if not (os.path.exists('train_data.npy') and os.path.exists('test_data.npy') \
        and os.path.exists('validation_data.npy')):
    def create_label(category):
        """ Create an one-hot encoded vector from image name """
        if category == 'Cat':
            return np.array([1,0])
        elif category == 'Dog':
            return np.array([0,1])
        return np.array([0, 0])

    def create_data(category, n):
        testing_data = []
        label = create_label(category)
    	# read images after the training images
        for i in tqdm(range(1, n+1)):
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
    nb_cats = min(count_images('Cat'), MAX_NB_IMAGES_PER_CATEGORY)
    nb_dogs = min(count_images('Dog'), MAX_NB_IMAGES_PER_CATEGORY)

    # read images
    cats_data = create_data('Cat', nb_cats)
    dogs_data = create_data('Dog', nb_dogs)
    data = cats_data + dogs_data

    # shuffle images
    shuffle(data)

    # split into train / validation / test subsets
    train_data = data[:-(NB_IMAGES_FOR_VALIDATION + NB_IMAGES_FOR_TEST)]
    validation_data = data[-(NB_IMAGES_FOR_VALIDATION + NB_IMAGES_FOR_TEST):-NB_IMAGES_FOR_TEST]
    test_data = data[-NB_IMAGES_FOR_TEST:]

    np.save('train_data.npy', train_data)
    np.save('validation_data.npy', validation_data)
    np.save('test_data.npy', test_data)
else:
    train_data = np.load('train_data.npy')
    validation_data = np.load('validation_data.npy')
    test_data = np.load('test_data.npy')

X_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train_data]

X_validation = np.array([i[0] for i in validation_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_validation = [i[1] for i in validation_data]

# CREATE NN

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

# TRAIN NN
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

model_already_trained = False
model_file_pattern = re.compile('^model.tfl')
for filename in os.listdir():
    if model_file_pattern.match(filename):
        model_already_trained = True
        break

if model_already_trained:
    model.load('model.tfl')
else:
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
        validation_set=({'input': X_validation}, {'targets': y_validation}),
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save("model.tfl")

# PREDICT
fig=plt.figure(figsize=(16, 12))

for num, data in enumerate(test_data[:16]):

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(4, 4, num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label='Dog'
    else:
        str_label='Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
