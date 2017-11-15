"""Fully Convolutional Neural Network
Architecture based on vgg16
"""
import keras.backend as K
import tensorflow as tf
from up_sampling import *
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from data_generator import SegDataGenerator
from os.path import expanduser



def softmax_crossentropy_ignoring_last_label(y_true, y_pred):
    """loss function"""
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean


# accuracy measurement
def accuracy_ignoring_last_label(y_true, y_pred):
    """ performance metric"""
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))


def lines_in_file(path):
    fp = open(path)
    lines = fp.readlines()
    fp.close()
    return len(lines)


class FCN_VGG16:

    def __init__(self, input_size=(224,224,3),
                 activation='relu',
                 weight_decay=0.0,
                 classes=21):
        """default setting for VOC dataset"""
        self.input_size = input_size
        # build model
        model = Sequential()

        # Block 1
        model.add(Conv2D(64, (3, 3), activation=activation,
                         padding='same', name='block1_conv1',
                         kernel_regularizer=l2(weight_decay),
                         input_shape=input_size))
        model.add(Conv2D(64, (3, 3), activation=activation,
                         padding='same', name='block1_conv2',
                         kernel_regularizer=l2(weight_decay)))
        model.add(MaxPooling2D((2,2), strides=(2,2), name="block1_pool"))

        # Block 2
        model.add(Conv2D(128, (3, 3), activation=activation,
                         padding='same', name='block2_conv1',
                         kernel_regularizer=l2(weight_decay)))
        model.add(Conv2D(128, (3, 3), activation=activation,
                         padding='same', name='block2_conv2',
                         kernel_regularizer=l2(weight_decay)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        model.add(Conv2D(256, (3, 3), activation=activation,
                         padding='same', name='block3_conv1',
                         kernel_regularizer=l2(weight_decay)))
        model.add(Conv2D(256, (3, 3), activation=activation,
                         padding='same', name='block3_conv2',
                         kernel_regularizer=l2(weight_decay)))
        model.add(Conv2D(256, (3, 3), activation=activation,
                         padding='same', name='block3_conv3',
                         kernel_regularizer=l2(weight_decay)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Block 4
        model.add(Conv2D(512, (3, 3), activation=activation,
                         padding='same', name='block4_conv1',
                         kernel_regularizer=l2(weight_decay)))
        model.add(Conv2D(512, (3, 3), activation=activation,
                         padding='same', name='block4_conv2',
                         kernel_regularizer=l2(weight_decay)))
        model.add(Conv2D(512, (3, 3), activation=activation,
                         padding='same', name='block4_conv3',
                         kernel_regularizer=l2(weight_decay)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        # Block 5
        model.add(Conv2D(512, (3, 3), activation=activation,
                         padding='same', name='block5_conv1',
                         kernel_regularizer=l2(weight_decay)))
        model.add(Conv2D(512, (3, 3), activation=activation,
                         padding='same', name='block5_conv2',
                         kernel_regularizer=l2(weight_decay)))
        model.add(Conv2D(512, (3, 3), activation=activation,
                         padding='same', name='block5_conv3',
                         kernel_regularizer=l2(weight_decay)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

        # fully convolutional
        model.add(Conv2D(4096, (7, 7), activation=activation,
                         padding='same', name='fc1',
                         kernel_regularizer=l2(weight_decay)))
        model.add(Dropout(0.5))
        model.add(Conv2D(4096, (1, 1), activation=activation,
                         padding='same', name='fc2',
                         kernel_regularizer=l2(weight_decay)))
        model.add(Dropout(0.5))

        # classification
        model.add(Conv2D(classes, (1, 1), kernel_initializer='he_normal',
                         activation='linear', padding='valid', strides=(1, 1),
                         kernel_regularizer=l2(weight_decay)))

        model.add(BilinearUpSampling2D(size=(32,32)))

        self.model = model
        self.classes = classes

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path, by_name=True)


    def train(self, learning_rate=0.001, epochs=100, resume_training=False):
        """"""
        # settings
        
        checkpoint_weights = 'seg_checkpoint_weights.hdf5'
        final_weights = 'fcn_final_weights.hdf5'
        train_file_path = expanduser('~/.keras/datasets/VOC2012/combined_imageset_train.txt')
        val_file_path = expanduser('~/.keras/datasets/VOC2012/combined_imageset_val.txt')
        data_folder = expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
        label_folder = expanduser('~/.keras/datasets/VOC2012/combined_annotations')
        data_suffix = '.jpg'
        label_suffix = '.png'
        ignore_label = 255
        label_cval = 255
        batch_size = 16
        target_size = self.input_size[:2]
        metrics = [accuracy_ignoring_last_label]
        opt = Adam(lr=learning_rate)
        # opt = SGD(lr=learning_rate, momentum=0.9)

        # compile
        self.model.compile(loss=softmax_crossentropy_ignoring_last_label,
                           optimizer=opt,
                           metrics=metrics)
        if resume_training:
            self.model.load_weights(checkpoint_weights, by_name=True)

        # data generator
        train_datagen = SegDataGenerator(
            zoom_range=[0.5, 2.0],
            zoom_maintain_shape=True,
            rotation_range=0.,
            shear_range=0,
            horizontal_flip=True,
            channel_shift_range=20.,
            fill_mode='constant',
            label_cval=label_cval)
        val_datagen = SegDataGenerator()
        train_generator = train_datagen.flow_from_directory(
            file_path=train_file_path,
            data_dir=data_folder, data_suffix=data_suffix,
            label_dir=label_folder, label_suffix=label_suffix,
            classes=self.classes,
            target_size=target_size, color_mode='rgb',
            batch_size=batch_size, shuffle=True,
            ignore_label=ignore_label)

        val_generator = val_datagen.flow_from_directory(
            file_path=val_file_path,
            data_dir=data_folder, data_suffix=data_suffix,
            label_dir=label_folder, label_suffix=label_suffix,
            classes=self.classes,
            target_size=target_size, color_mode='rgb',
            batch_size=batch_size, shuffle=False)

        # checkpoint
        checkpoint = ModelCheckpoint(checkpoint_weights,
                                     verbose=1, save_best_only=True)
        callbacks = [checkpoint]

        # training
        history = self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=lines_in_file(train_file_path) // batch_size,
            validation_steps=lines_in_file(val_file_path) // batch_size,
            epochs=epochs,
            workers=4,
            validation_data=val_generator,
            verbose=1, callbacks=callbacks
            )
        self.model.save_weights(final_weights)

def main():
    """"""
    fcn = FCN_VGG16()
    print(fcn.model.summary())
    # load pretrained
    pretrained_weights = 'fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    fcn.load_weights(pretrained_weights)
    fcn.train(learning_rate=0.001, resume_training=False)


if __name__ == '__main__':
    main()
