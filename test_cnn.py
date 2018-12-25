import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization

from sequence import trainSequence



length = 128
width = 128
channels = 3
batch_size = 25
epochs = 3




def main():

    model = Sequential()
    model.add(Conv2D(5, (3, 3), padding='same', input_shape=[length,width,channels]))
    model.add(Activation('relu'))

    model.add(Conv2D(6, (3, 3), padding='same', input_shape=[length,width,channels]))
    model.add(Activation('relu'))
    
    model.add(Conv2D(7, (5, 5), padding='same', input_shape=[length,width,channels]))
    model.add(Activation('relu'))

    model.add(Conv2D(7, (7, 7), padding='same', input_shape=[length,width,channels]))
    model.add(Activation('relu'))

    model.add(Conv2D(7, (11, 11), padding='same', input_shape=[length,width,channels]))
    model.add(Activation('relu'))

    model.add(Conv2D(7, (15, 15), padding='same', input_shape=[length,width,channels]))
    model.add(Activation('relu'))

    model.add(Conv2D(7, (11, 11), padding='same', input_shape=[length,width,channels]))
    model.add(Activation('relu'))

    model.add(Conv2D(7, (7, 7), padding='same', input_shape=[length,width,channels]))
    model.add(Activation('relu'))

    model.add(Conv2D(7, (5, 5), padding='same', input_shape=[length,width,channels]))
    model.add(Activation('relu'))

    model.add(Conv2D(6, (3, 3), padding='same', input_shape=[length,width,channels]))
    model.add(Activation('relu'))

    model.add(Conv2D(5, (3, 3), padding='same', input_shape=[length,width,channels]))
    model.add(Activation('relu'))



    #output has 3 channels and no nonlinearity
    model.add(Conv2D(3, (3, 3), padding='same'))





    # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    # opt = keras.optimizers.rmsprop(lr=0.0005, decay=1e-6)
    opt = keras.optimizers.adam(lr=0.0005, decay=1e-6)
    # opt = keras.optimizers.SGD(lr=0.0005, decay=1e-6)

    loss = 'mean_squared_error'
    #loss = 'mean_absolute_error'


    model.compile(loss=loss,
              optimizer=opt,
              metrics=['mae', 'mse'])


    train_data = trainSequence(batch_size)
    model.fit_generator(train_data,
              epochs=epochs,
              shuffle=True)


    model.save('test_model.h5')




if __name__ == '__main__':
    main()




