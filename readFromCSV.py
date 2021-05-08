import csv
import tensorflow as tf
import pandas as pd
import numpy as np

# Code snippet taken from https://stackoverflow.com/questions/43697240/how-can-i-split-a-dataset-from-a-csv-file-for-training-and-testing

data_file = pd.read_csv("aug_train.csv")

data_file = pd.get_dummies(data_file)

tf.keras.utils.to_categorical(data_file.target)  # TODO: CHeck if it matters

data_file.to_csv("clean_data.csv")

print(data_file.info())
print(data_file.isnull().any())

train_split = np.random.rand(len(data_file)) <= .7
ds_train = data_file[train_split]
# Method for splitting data sets inspired from-> https://stackoverflow.com/questions/43697240/how-can-i-split-a-dataset-from-a-csv-file-for-training-and-testing
ds_other = data_file[~train_split]
val_split = np.random.rand(len(ds_other)) <= .5
ds_test = ds_other[val_split]
ds_val = ds_other[~val_split]

train_target = ds_train.pop('target')
val_target = ds_val.pop("target")

data_numpy = ds_train.to_numpy()

train_target = pd.get_dummies(train_target)
print(train_target)

data_tensor = tf.convert_to_tensor(data_numpy)

print(len(data_tensor))
print(len(ds_test))
print(len(ds_val))


def get_compiled_model():
    model = tf.keras.Sequential([

        tf.keras.layers.Dense(63, activation='tanh'),
        tf.keras.layers.Dense(140, activation='tanh'),
        #tf.keras.layers.Dense(63, activation='tanh'),
        tf.keras.layers.Dense(2, activation='softmax'),

    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model


model = get_compiled_model()

model.fit(data_tensor, train_target, epochs=100, validation_data=(ds_val, val_target), shuffle='true')

test_target = ds_test.pop('target')
#
test_target = pd.get_dummies(test_target)

model.evaluate(ds_test, test_target)

prediction = model.predict(ds_test)

prediction = pd.DataFrame(prediction)
prediction.to_csv("predictions.csv")
