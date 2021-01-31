from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D

from tensorflow.keras.layers import Lambda, Flatten

def alexnet(input_shape, num_classes, hscale=1, dense_neurons=512):
    model = Sequential()
    model.add(Conv2D(32 * hscale, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32 * hscale, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64 * hscale, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64 * hscale, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(dense_neurons))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.summary()
    return model

