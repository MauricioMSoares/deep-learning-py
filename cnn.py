# Convolutional Neural Network

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense


def setup_cnn(x_train, x_test, y_train, y_test):
    # Loading data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Re-shaping data
    x_train = x_train.reshape((x_train.shape[0]), x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape((x_test.shape[0]), x_test.shape[1], x_test.shape[2], 1)

    # Checking shape after re-shaping
    print(x_train.shape)
    print(y_train.shape)

    # Normalizing pixel values
    x_train = x_train / 255
    x_test = x_test / 255

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPool2D(2, 2))

    # Adding fully connected layer
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))

    # Adding output layer
    model.add(Dense(10, activation="softmax"))

    # Compiling the model
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Fitting the model
    model.fit(x_train, y_train, epochs=10)

    # Evaluating the model
    model.evaluate(x_test, y_test)
