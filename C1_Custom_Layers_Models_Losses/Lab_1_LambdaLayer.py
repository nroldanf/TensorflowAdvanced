'''
This lab will show how you can define custom layers with the 
[Lambda](https://keras.io/api/layers/core_layers/lambda/) layer. 
You can either use [lambda functions](https://www.w3schools.com/python/python_lambda.asp) 
within the Lambda layer or define a custom function that the Lambda layer will call. Let's get started!
'''
import tensorflow as tf
from tensorflow.keras import backend as K

def my_relu(x):
    return K.maximum(-0.1, x)

if __name__ == '__main__':

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    # Normalize images
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Crate the model
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Lambda(lambda x: tf.abs(x)), 
    tf.keras.layers.Dense(10, activation='softmax')
    ])
    # Compile and train the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)

    # Another way to use the Lambda layer is to pass in a function 
    # defined outside the model. The code below shows how a custom ReLU function 
    # is used as a custom layer in the model. 

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Lambda(my_relu), 
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)