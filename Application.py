from __future__ import print_function

import argparse
import keras
from keras.datasets import cifar10, cifar100, mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import cv2
# from resnet_model import resnet_v1, lr_schedule, resnet_bilinear

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import datetime, csv
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.callbacks import TensorBoard

import timeit

from linear.alexnet import alexnet
from linear.resnet import resnet_v1
from quadratic.alexnet import qalexnet
from quadratic.resnet import qresnet_v1

def main(model_name, dataset, neural_function, batch_size=32, epochs=100, color=1, data_augmentation=True, is_neg=1):
    output_folder = "Output"
    save_models = "saved_models/"

    save_dir = os.path.join(output_folder, save_models)

    if dataset == "cifar10":
        (x_train_c, y_train), (x_test_c, y_test) = cifar10.load_data()
        num_classes = 10
    elif dataset == "cifar100":
        (x_train_c, y_train), (x_test_c, y_test) = cifar100.load_data()
        num_classes = 100

    print("Loaded {} dataset".format(dataset))

    original_shape_train = x_train_c.shape
    original_shape_test = x_test_c.shape

    if color:
        x_train = x_train_c
        x_test = x_test_c
    else:

        x_train = np.zeros((original_shape_train[0], original_shape_train[1], original_shape_train[2], 1))
        for k in range(original_shape_train[0]):
            x_train[k] = np.expand_dims(cv2.cvtColor(x_train_c[k], cv2.COLOR_BGR2GRAY), axis=3)

        x_test = np.zeros((original_shape_test[0], original_shape_test[1], original_shape_test[2], 1))
        for k in range(original_shape_test[0]):
            x_test[k] = np.expand_dims(cv2.cvtColor(x_test_c[k], cv2.COLOR_BGR2GRAY), axis=3)

    print("Using {} images to train data".format("color" if color else "greyscale"))

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    input_shape = x_train.shape[1:]

    if neural_function == "linear":
        print("Using {} function for neurons".format(neural_function))
        if model_name == "alexnet":
            original_lr = 0.0001
            opt = RMSprop(lr=original_lr, decay=1e-6)
            dense_neu = 512
            model = alexnet(input_shape, num_classes, dense_neurons=dense_neu)
        elif model_name == "resnet":
            depth = 20
            model = resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)
            # opt = Adam(lr=lr_schedule(0))
            original_lr = 0.0001
            opt = RMSprop(lr=original_lr, decay=1e-6)

    elif neural_function == "quadratic":
        print("Using {} function for neurons".format(neural_function))
        if model_name == "alexnet":
            original_lr = 0.0001
            opt = RMSprop(lr=original_lr, decay=1e-6)
            model = qalexnet(input_shape, num_classes, activation='elu', is_neg=is_neg)
        elif model_name == "resnet":
            depth = 20
            model = qresnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes, q=True, is_neg=is_neg)
            original_lr = 0.0001
            opt = RMSprop(lr=original_lr, decay=1e-6)

    else:
        print("Undefined neural function \"{}\"".format(neural_function))

    print("Compiling {} model with {} neural function to train on {}".format(model_name, neural_function, dataset))
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    this_model_name = "{}_{}_{}_{}_{}".format(dataset, model_name, "negative" if is_neg else "positive",
                                              neural_function, "color" if color else "greyscale")
    print("Model name {}".format(this_model_name))

    tensor_log_dir = 'logs/{}'.format(this_model_name)
    output_tensor_logs = os.path.join(output_folder, tensor_log_dir)
    print("Tensorflow log directory - {}".format(output_tensor_logs))
    tensorboard = TensorBoard(log_dir=output_tensor_logs)

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        datagen.fit(x_train)

        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=epochs,
                            validation_data=datagen.flow(x_test, y_test),
                            workers=4, callbacks=[tensorboard])


        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, this_model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)

        scores = model.evaluate(x_test, y_test, verbose=1)

        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        results = [datetime.datetime.now(), this_model_name, model.count_params(), scores[0], scores[1]]

        testing_output_file = os.path.join(output_folder, "results.csv")
        with open(testing_output_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument('-m', '--model', type=str, help='Specify model to train [alexnet]', required=True)
    parser.add_argument('-d', '--dataset', type=str, help='Specify dataset [cifar10, cifar100]', required=True)
    parser.add_argument('-n', '--neural-function', type=str, help='Specify neural function [linear, quad]',
                        required=True)
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Specify batch size, default 32')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Specify number of epochs, default 100')
    parser.add_argument('-c', '--color', type=int, default=1,
                        help='Specify 1/0 for color/greyscale, default is 1')

    parser.add_argument('-da', '--data-augmentation', type=bool, default=True,
                        help='Specify True/False for data agumentation, default True')

    args = parser.parse_args()

    main(args.model, args.dataset, args.neural_function, args.batch_size, args.epochs, args.color,
         args.data_augmentation)
