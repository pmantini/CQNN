from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Reshape, \
    AveragePooling2D

from tensorflow.keras.models import Model

from qconv2d import QConv2d

from tensorflow.keras.layers import Lambda, Flatten
from keras import backend as K
from numpy import newaxis

QMaxPool2d = MaxPooling2D

def qalexnet(input_shape, num_classes, activation="elu", is_neg=1):
    input_img = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))

    layer_1 = QConv2d(32, 3, 1, padding='SAME', is_neg=is_neg)(input_img)
    act_1 = Activation(activation)(layer_1)
    layer_2 = QConv2d(32, 3, 1, padding='SAME', is_neg=is_neg)(act_1)
    act_2 = Activation(activation)(layer_2)
    pooling_2 = QMaxPool2d(2, padding='SAME')(act_2)
    dropout_1 = Dropout(0.25)(pooling_2)

    layer_3 = QConv2d(64, 3, 1, padding='SAME', is_neg=is_neg)(dropout_1)
    act_3 = Activation(activation)(layer_3)
    layer_4 = QConv2d(64, 3, 1, padding='SAME', is_neg=is_neg)(act_3)
    act_4 = Activation(activation)(layer_4)
    pooling_3 = QMaxPool2d(2, padding='SAME')(act_4)
    dropout_2 = Dropout(0.25)(pooling_3)

    flat = Flatten()(dropout_2)
    dense_1 = Dense(512)(flat)
    act_final = Activation(activation)(dense_1)
    dropout_final = Dropout(0.5)(act_final)
    final_dense = Dense(num_classes)(dropout_final)
    final = Activation('softmax')(final_dense)

    model = Model(inputs=input_img, outputs=final)
    model.summary()

    return model

