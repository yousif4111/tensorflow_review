import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


(X_train,y_train),(X_test,y_test) = mnist.load_data()

print(X_train.shape)
print(y_train.shape)

X_train = X_train.reshape(-1, 28 * 28).astype("float32")# / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype("float32")# / 255.0

# model = keras.Sequential(
#     [
#         keras.Input(shape=(28 * 28)),
#         layers.Dense(512, activation='relu'),
#         layers.Dense(256, activation='relu'),
#         layers.Dense(10)
#     ]
# )
#
# print(model.summary())
#
# # another way to create neural network
#
# model = keras.Sequential()
# model.add(keras.Input(shape=28 * 28))
# print(model.summary())
# model.add(layers.Dense(512, activation='relu'))
# print(model.summary())
# model.add(layers.Dense(256, activation = 'relu'))
# print(model.summary())
# model.add(layers.Dense(10))


# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(learning_rate=0.001),
#     metrics=["accuracy"]
# )
#
# model.fit(X_train, y_train, batch_size=32, epochs=5)
# model.evaluate(X_test, y_test, batch_size=32)

# # Functional API (A bit more flexible)
inputs = keras.Input(shape=28 * 28)
x = layers.Dense(512, activation='relu',name="First_hidden_layer")(inputs)
x = layers.Dense(256, activation='relu',name="Second_hidden_layer")(x)
outputs = layers.Dense(10, activation='softmax',name="Mnist_output")(x)
model= keras.Model(inputs=inputs,outputs=outputs)
print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer = keras.optimizers.Adamax(lr=0.001),
    metrics=["accuracy"],
)

model.fit(X_train,y_train,batch_size=32, epochs=5)
model.evaluate(X_test,y_test)

# # How to debug each layer one by one
# model = keras.Sequential()
# model.add(keras.Input(shape=28 * 28))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(256, activation = 'relu', name='my_layer'))
# model.add(layers.Dense(10))
#
# # first methods:
# model=keras.Model(inputs=model.inputs,
#                   outputs=[model.layers[-2].output] # model.layers[#] replace # by -1 to get output layer and -2 to get the layer before that and so on
#                   )
# # Second methods:
# model=keras.Model(inputs=model.inputs,
#                   outputs=[model.get_layer('my_layer').output])
# feature = model.predict(X_train)
# print(f"Single layer output {feature.shape}")
#
# # Multiple output
# model=keras.Model(inputs=model.input,
#                   outputs=[layer.output for layer in model.layers])
# features = model.predict(X_train)
#
# for feature in features:
#     print(f"All layer output together: {feature.shape}")


# # Functional API (A bit more flexible)
# inputs = keras.Input(shape=28 * 28)
# x = layers.Dense(512, activation='relu',name="First_hidden_layer")(inputs)
# x = layers.Dense(256, activation='relu',name="Second_hidden_layer")(x)
# outputs = layers.Dense(10, activation='softmax',name="Mnist_output")(x)
# model= keras.Model(inputs=inputs,outputs=outputs)
# print(model.summary())
#
#
# # list_optimizer = ['SGD', 'RMSprop', 'Adam', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']
# #
# # highest_accuracy = 0.0
# # best_optimizer = ""
# # for opti in list_optimizer:
# #     model.compile(
# #         loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
# #         optimizer = opti,
# #         metrics=["accuracy"],
# #     )
# #
# #     model.fit(X_train,y_train,batch_size=32, epochs=1)
# #     _, accur= model.evaluate(X_test,y_test)
# #     if accur>highest_accuracy:
# #         highest_accuracy=accur
# #         best_optimizer=opti
# # print("Optimizer with the highest starting accuracy:", best_optimizer)
# #
# #
# #




