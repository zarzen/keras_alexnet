import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop


class AlexNet:
    def __init__(self,
                 num_classes=1000,
                 lr_rate=0.001,
                 dropout_keep_prob=0.5):
        """"""
        input_shape = (227, 227, 3)
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

        model.add(Dense(num_classes, activation='relu'))
        model.add(BatchNormalization())

        model.add(Activation('softmax'))

        # model compile
        optimizer = RMSprop(lr = lr_rate)
        model.compile(optimizer=optimizer,
                      loss='mse',
                      metrics=['accuracy'])

        self.model = model

    def train(self, data_folder, batch_size=32 ):
        """"""

if __name__ == '__main__':
    net = AlexNet()
    print(net.model.summary())


