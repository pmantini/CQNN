from tensorflow.keras import Model
from qconv2d import QConv2d
from tensorflow.keras.layers import Dense, Activation, AveragePooling2D, Input, Flatten, Add, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization as QBatchNormalization

QMaxPool2d = MaxPooling2D
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def qresnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='elu',
                 batch_normalization=True,
                 conv_first=True, hscale=1, is_neg=1, q_function="naive"):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """

    conv = QConv2d(num_filters,
                  kernel_size=kernel_size,
                  stride=strides,
                  padding='SAME', is_neg=is_neg)

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = QBatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = QBatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def qresnet_v1(input_shape, depth, num_classes=10, q = False, batch_normalization=True, hscale=1, is_neg=1,
              include_top=True, q_function = "naive"):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """


    res_layer = qresnet_layer

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = res_layer(inputs=inputs, batch_normalization=batch_normalization, hscale=hscale, is_neg=is_neg, q_function=q_function)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = res_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides, batch_normalization=batch_normalization, hscale=hscale, is_neg=is_neg, q_function=q_function)
            y = res_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None, batch_normalization=batch_normalization, hscale=hscale, is_neg=is_neg, q_function=q_function)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = res_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False, hscale=hscale, is_neg=is_neg, q_function=q_function)
            # x = keras.layers.add([x, y])
            x = Add()([x, y])
            if q:
                x = Activation('elu')(x)
            else:
                x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    if include_top:

        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)
    else:
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = y
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

