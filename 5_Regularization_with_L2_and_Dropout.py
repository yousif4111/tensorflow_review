import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10, mnist
import time

# ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± #
# Regularization with L2 and Dropout #
# ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±± #

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape)
print(y_train.shape)
print(len(X_train))

# let's start with normalizing the data
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# over come the overfitting problem we there are different methods
# I will examine regularization technique from tensorflow to apply
# extra penalties to overcome overfitting problem
# from tensorflow.keras import regularizers


# to add the L2 regularizer we need to add it to each layer separately
def my_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    # adding regularization for fully connected layer as well
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    # Adding Dropout layer to 0.5 of connection between previous and next layer
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = my_model()

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"]
)
start_time = time.time()
model.fit(X_train, y_train, batch_size=64, epochs=1)
model.evaluate(X_test, y_test, batch_size=64, verbose=2)
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time
print("Training Time:", training_time, "seconds")
