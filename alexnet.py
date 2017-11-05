import keras
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


class AlexNet:
    def __init__(self,
                 num_classes=1000,
                 dropout_keep_prob=0.5):
        """"""
        input_shape = (227, 227, 3)
        self.input_shape = input_shape
        model = Sequential()
        # conv1
        model.add(Conv2D(96, (11,11), strides=4, padding='valid', activation='relu',
                         input_shape=input_shape))
        model.add(BatchNormalization())
        # pool1
        model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='valid'))

        # conv2
        model.add(ZeroPadding2D((2,2)))
        model.add(Conv2D(256, (5,5), padding='valid', activation='relu'))
        model.add(BatchNormalization())
        # pool2
        model.add(MaxPooling2D(pool_size=(3,3), strides=2))

        # conv3
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(384, (3,3), padding='valid', activation='relu'))
        model.add(BatchNormalization())

        # conv4
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(384, (3,3), padding='valid', activation='relu'))
        model.add(BatchNormalization())

        # conv5
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3,3), padding='valid', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((3,3), strides=2))

        # full connected layer
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_keep_prob))

        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_keep_prob))

        model.add(Dense(num_classes))
        model.add(BatchNormalization())

        model.add(Activation('softmax'))

        self.model = model


    def train(self, train_data_path, val_data_path,
              batch_size=32, n_epoch=100,
              learning_rate=0.001):
        """"""
        train_generator = self._get_data_generator(train_data_path, True, batch_size)
        validation_generator = self._get_data_generator(val_data_path, False, 10)

        # model compile
        opt = RMSprop(lr = learning_rate, decay=1e-6)
        sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

        # check point
        model_weights_path = 'weights.best.hdf5'
        checkpoint = ModelCheckpoint(model_weights_path, monitor='val_acc',
                                     verbose=1, save_best_only=True, mode='max')
        callback_list = [checkpoint]

        train_history = self.model.fit_generator(
            train_generator,
            steps_per_epoch=int(40000/batch_size),
            validation_data=validation_generator,
            validation_steps=100,
            epochs=n_epoch,
            verbose=1,
            callbacks=callback_list)

        return train_history

    def save_model(self, save_dir, model_name):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        self.model.save(model_path)



    def _get_data_generator(self, data_path, is_train, batch_size):
        if is_train:
            datagen = ImageDataGenerator(
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)
        else:
            datagen = ImageDataGenerator()

        data_generator = datagen.flow_from_directory(
            data_path,
            batch_size=batch_size,
            shuffle=True,
            target_size=self.input_shape[:-1],
            class_mode='categorical')
        return data_generator


if __name__ == '__main__':
    net = AlexNet()
    print(net.model.summary())
    trained_weights = 'weights.best.hdf5'
    if os.path.exists(trained_weights):
        net.model.load_weights(trained_weights, by_name=True)

    net.train('./data/train', './data/validation', batch_size=64, learning_rate=0.001)
    net.save_model('./', 'trained_alexnet')

