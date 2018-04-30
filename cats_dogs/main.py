import cv2
import numpy as np
import os
from random import shuffle
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tqdm import tqdm
import math
import re

# 12500 images available per category
MAX_NB_IMAGES_PER_CATEGORY = 10000
NB_IMAGES_FOR_VALIDATION = 500
NB_IMAGES_FOR_TEST = 100

DATASET_DIR = 'dataset/PetImages'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'dogs-vs-cats-convnet'

# check if datasets exists
if not (os.path.exists('train_data.npy') and os.path.exists('test_data.npy') \
        and os.path.exists('validation_data.npy')):
    def create_label(category):
        """ Create an one-hot encoded vector from category name """
        if category == 'Cat':
            return np.array([1,0])
        elif category == 'Dog':
            return np.array([0,1])
        return np.array([0, 0])

    def create_data(category, n):
	# read n images for the given category
        data = []
        label = create_label(category)
    	# read images after the training images
        for i in tqdm(range(1, n+1)):
            path = os.path.join(DATASET_DIR + '/' + category, str(i) + '.jpg')
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # check if the image is valid
            if img_data is not None:
                img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
                # resized image + his label (one hot encoded vector) + original image path
                data.append([np.array(img_data), label, path])
        shuffle(data)
        return data

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

    # save the datasets
    np.save('train_data.npy', train_data)
    np.save('validation_data.npy', validation_data)
    np.save('test_data.npy', test_data)
else:
    # load the datasets
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

# check if the model has already been trained
model_already_trained = False
model_file_pattern = re.compile('^model.tfl')
for filename in os.listdir():
    if model_file_pattern.match(filename):
        model_already_trained = True
        break

# if there is a trained model, load it
# else, train the model and save it
if model_already_trained:
    model.load('model.tfl')
else:
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
        validation_set=({'input': X_validation}, {'targets': y_validation}),
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save("model.tfl")

# PREDICT
X_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test_data]

count_good_predictions = 0
for i in range(len(X_test)):

    # data[-1][0] == 1 => cat
    # data[-1][1] == 1 => dog

    data = X_test[i].reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]
    if np.argmax(model_out) == np.argmax(y_test[i]):
       	count_good_predictions += 1
    else:
        print("Wrong prediction for %s" % (test_data[i][2]))
        print("prediction value: %f" % (max(model_out)))

print("Predictions: %d on %d" % (count_good_predictions, len(X_test)))
