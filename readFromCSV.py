
import csv
import tensorflow as tf
import pandas as pd
import numpy as np

# Code snippet taken from https://stackoverflow.com/questions/43697240/how-can-i-split-a-dataset-from-a-csv-file-for-training-and-testing

data_file = pd.read_csv("aug_train.csv")

data_file['gender'] = pd.Categorical(data_file['gender'])
data_file['gender'] = data_file.gender.cat.codes

data_file['relevant_experience'] = pd.Categorical(data_file['relevant_experience'])
data_file['relevant_experience'] = data_file.relevant_experience.cat.codes

data_file['enrolled_university'] = pd.Categorical(data_file['enrolled_university'])
data_file['enrolled_university'] = data_file.enrolled_university.cat.codes

data_file['education_level'] = pd.Categorical(data_file['education_level'])
data_file['education_level'] = data_file.education_level.cat.codes

data_file['major_discipline'] = pd.Categorical(data_file['major_discipline'])
data_file['major_discipline'] = data_file.major_discipline.cat.codes

data_file['experience'] = pd.Categorical(data_file['experience'])
data_file['experience'] = data_file.experience.cat.codes

data_file['company_size'] = pd.Categorical(data_file['company_size'])
data_file['company_size'] = data_file.company_size.cat.codes

data_file['company_type'] = pd.Categorical(data_file['company_type'])
data_file['company_type'] = data_file.company_type.cat.codes

data_file['last_new_job'] = pd.Categorical(data_file['last_new_job'])
data_file['last_new_job'] = data_file.last_new_job.cat.codes

data_file.to_csv("clean_data.csv")

train_split = np.random.rand(len(data_file)) <= .7
ds_train = data_file[train_split]
# Method for splitting data sets inspired from-> https://stackoverflow.com/questions/43697240/how-can-i-split-a-dataset-from-a-csv-file-for-training-and-testing
ds_other = data_file[~train_split]
val_split = np.random.rand(len(ds_other)) <= .5
ds_test = ds_other[val_split]
ds_val = ds_other[~val_split]

train_target = ds_train.pop('target')

dataset = tf.data.Dataset.from_tensor_slices((ds_train.values, train_target.values))



dataset = dataset.shuffle(len(ds_train)).batch(1)

# def get_compiled_model():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(11, activation='relu'),
#         tf.keras.layers.Dense(11, activation='relu'),
#         tf.keras.layers.Dense(1),
#         tf.keras.layers.Dense(1)
#     ])
#
#     model.compile(optimizer='adam',
#                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#                   metrics=['accuracy'])
#     return model
#
#
# model = get_compiled_model()
# model.fit(dataset, epochs=5)

# model.save('saved_model/my_model')
new_model = tf.keras.models.load_model('saved_model/my_model')
test_target = ds_test.pop('target')


new_model.evaluate(ds_test, test_target)



val_target=ds_val.pop('target')

new_model.evaluate(ds_val, val_target)

#
# model.evaluate(dataset_test)
