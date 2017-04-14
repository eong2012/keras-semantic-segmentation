"""
ResNet based FCN.
"""
from keras.models import Model
from keras.layers import (
    Input, Activation, Reshape, Conv2D, Lambda, Add)
import tensorflow as tf

from .resnet50 import ResNet50


FCN_RESNET = 'fcn_resnet'


def make_fcn_resnet(input_shape, nb_labels, use_pretraining, freeze_base):
    nb_rows, nb_cols, _ = input_shape
    input_tensor = Input(shape=input_shape)
    weights = 'imagenet' if use_pretraining else None

    base_model = ResNet50(
        include_top=False, weights=weights, input_tensor=input_tensor)

    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False

    # Get final 32x32, 16x16, and 8x8 layers in the original
    # ResNet by that layers's name. These are the "skip connections."
    x32 = base_model.get_layer('act3d').output
    x16 = base_model.get_layer('act4f').output
    x8 = base_model.get_layer('act5c').output

    # Compress each skip connection so it has nb_labels channels.
    c32 = Conv2D(nb_labels, (1, 1), name='conv_labels_32')(x32)
    c16 = Conv2D(nb_labels, (1, 1), name='conv_labels_16')(x16)
    c8 = Conv2D(nb_labels, (1, 1), name='conv_labels_8')(x8)

    # Resize each compressed skip connection using bilinear interpolation.
    # This operation isn't built into Keras, so we use a LambdaLayer
    # which allows calling a Tensorflow operation.
    def resize_bilinear(images):
        return tf.image.resize_bilinear(images, [nb_rows, nb_cols])

    r32 = Lambda(resize_bilinear, name='resize_labels_32')(c32)
    r16 = Lambda(resize_bilinear, name='resize_labels_16')(c16)
    r8 = Lambda(resize_bilinear, name='resize_labels_8')(c8)

    # Merge the three layers together using summation.
    m = Add(name='merge_labels')([r32, r16, r8])

    # Pass it through softmax to get probabilities. We need to reshape
    # and then un-reshape because Keras expects input to softmax to
    # be 2D.
    x = Reshape((nb_rows * nb_cols, nb_labels))(m)
    x = Activation('softmax')(x)
    x = Reshape((nb_rows, nb_cols, nb_labels))(x)

    model = Model(inputs=input_tensor, outputs=x)

    return model
