import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10,mnist
import time

#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±#
#       Dataset: CIFAR-10     #
#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±#
"""
CIFAR-10 is a widely used dataset in computer vision.
It contains 60,000 color images of 10 different objects.
The dataset is split into a training set of 50,000 images 
and a test set of 10,000 images. It's commonly used for image classification
tasks and evaluating machine learning models.
"""


(X_train,y_train),(X_test,y_test) = cifar10.load_data()
print(X_train.shape)
print(y_train.shape)
print(len(X_train))

# (X_train,y_train),(X_test,y_test) = mnist.load_data()
# print(X_train.shape)
# print(y_train.shape)
#
# print(X_train[0,0,0])

"""
We can notice differnce between minsset and cifar10 in the extra dimension that
describe color red, blue, green.
"""

# let's start with normalizing the data
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±#
#       Sequential Model    #
#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±#

# build Sequentail model
# model = keras.Sequential(
#     [
#         # initaiting the input layers with dimension of data
#         keras.Input(shape=(32,32,3)),
#         # Start with basic convlosional layer
#         # with 32 and kernal (filter) 3 hight and width
#         layers.Conv2D(32, 3, padding='valid', activation='relu'),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         # let's create more Conv and pooling layers
#         layers.Conv2D(64, 3, activation = 'relu'),
#         layers.MaxPooling2D(),
#         layers.Conv2D(128, 3, activation = 'relu'),
#         # the multidimensional output from the previous layers
#         # into a one-dimensional array using Flatten layer.
#         layers.Flatten(),
#         layers.Dense(64, activation = 'relu'),
#         # Output layer
#         layers.Dense(10),
#     ]
# )
"""     
Padding in CNN
In convolutional neural networks (CNNs), padding refers
 to the technique of adding extra pixels to the input image 
 or feature map before performing convolution operations. 
 The purpose of padding is to preserve spatial dimensions and 
 prevent information loss during the convolutional layers.
"""
# print(model.summary())
#
# # let's complie the model:
# # Categorical Cross-Entropy loss is commonly used for multi-class
# # classification problems, where each input can belong to only one c
# model.compile(
#     loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer = keras.optimizers.Adam(learning_rate = 3e-4),
#     metrics = ["accuracy"]
# )
#
# model.fit(X_train, y_train, batch_size=64, epochs=10)
# model.evaluate(X_test, y_test, batch_size=64, verbose=2)

#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±#
#     Functional API Model    #
#±±±±±±±±±±±±±±±±±±±±±±±±±±±±±#

# Batch Normalization: Is normalization technique that is applied
# inside the neural network between layer before inputing to activation
# function based on batch input to improve traning
# stability and spped. It also allows sub-optimal starts.
#
def my_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, padding='valid')(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = my_model()



print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer= keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"]
)
start_time = time.time()
model.fit(X_train, y_train, batch_size=64, epochs=1)
model.evaluate(X_test, y_test, batch_size=64, verbose=2)
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time
print("Training Time:", training_time, "seconds")


