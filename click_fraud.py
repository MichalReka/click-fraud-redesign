import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
import numpy as np
import dask as d
import dask.dataframe as dd
from tensorflow import feature_column

def df_to_dataset(dataframe, shuffle=True, batch_size=32, labels=True):
    dataframe = dataframe.copy()
    if labels:
        labels = dataframe.pop('is_attributed')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds 
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint32',
        'device'        : 'uint32',
        'os'            : 'uint32',
        'channel'       : 'uint32',
        'click_time'    : 'object',
        'is_attributed' : 'uint8',
        }
#Dask znacznie przyspiesza wczytywanie csv
dd_train = dd.read_csv('train.csv',dtype=dtypes, usecols=['ip','app','device','os','channel','click_time','is_attributed'])
#zamiana str na object datetime64
dd_train['click_time']=dd.to_datetime(dd_train['click_time'], format='%Y-%m-%d %H:%M:%S')
#compute konwertuje na dataframe pandas - latwiejsza manipulacja danymi, wspolpraca z keras
df_train = dd_train[dd_train.is_attributed == 1] 
df_train=df_train.compute()
df_train_rest = dd_train[dd_train.is_attributed == 0] 
df_train_rest = df_train_rest.sample(frac=0.004)
df_train_rest = df_train_rest.compute()
df_train = train_df = pd.concat([df_train, df_train_rest])
df_train['day'] = df_train['click_time'].dt.day.astype(int)
df_train['hour'] = df_train['click_time'].dt.hour.astype(int)
df_train['minute'] = df_train['click_time'].dt.minute.astype(int)
df_train['second'] = df_train['click_time'].dt.second.astype(int)
df_train.drop(['click_time'], axis='columns', inplace=True)
feature_columns = []

# numeric cols
for col in ['ip']:
    feature_columns.append(feature_column.numeric_column(col))

# embedding columns
app = feature_column.categorical_column_with_vocabulary_list(
    'app', df_train.app.unique())
app_embedding = feature_column.embedding_column(app, dimension=64)
feature_columns.append(app_embedding)

os = feature_column.categorical_column_with_vocabulary_list(
    'os', df_train.os.unique())
os_embedding = feature_column.embedding_column(os, dimension=32)
feature_columns.append(os_embedding)

device = feature_column.categorical_column_with_vocabulary_list(
    'device', df_train.device.unique())
device_embedding = feature_column.embedding_column(device, dimension=32)
feature_columns.append(device_embedding)

channel = feature_column.categorical_column_with_vocabulary_list(
    'channel', df_train.channel.unique())
channel_embedding = feature_column.embedding_column(channel, dimension=32)
feature_columns.append(channel_embedding)

day = feature_column.categorical_column_with_vocabulary_list(
    'day', df_train.day.unique())
day_embedding = feature_column.embedding_column(day, dimension=8)
feature_columns.append(day_embedding)

hour = feature_column.categorical_column_with_vocabulary_list(
    'hour', df_train.hour.unique())
hour_embedding = feature_column.embedding_column(hour, dimension=8)
feature_columns.append(hour_embedding)

minute = feature_column.categorical_column_with_vocabulary_list(
    'minute', df_train.minute.unique())
minute_embedding = feature_column.embedding_column(minute, dimension=8)
feature_columns.append(minute_embedding)

second = feature_column.categorical_column_with_vocabulary_list(
    'second', df_train.second.unique())
second_embedding = feature_column.embedding_column(second, dimension=8)
feature_columns.append(second_embedding)
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
from sklearn.model_selection import train_test_split
df_train, df_val = train_test_split(df_train, stratify=df_train['is_attributed'], test_size=0.2, shuffle=True)
ds_train = df_to_dataset(df_train, batch_size=256).prefetch(tf.data.experimental.AUTOTUNE)
ds_val = df_to_dataset(df_val, shuffle=False, batch_size=256).prefetch(tf.data.experimental.AUTOTUNE)


tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA - szybszego kompilatora

model= Sequential(feature_layer)
for i in range(10):
    model.add(Dense(units=150, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))
#Binary Classification Entropy (BCE) - output to przedzial od 0 do 1 - czyli sigmoid, optimizer - default
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='bce')
#parametry przystosowane do mozliwosci mojego komputera
model.fit(ds_train, validation_data=ds_val, epochs=15,workers=6)   #bez batch_size - input to dataset, batches nie jest wspierane


ddata = dd.read_csv('test.csv',dtype=dtypes)
ddata['click_time']=dd.to_datetime(ddata['click_time'], format='%Y-%m-%d %H:%M:%S')
df = d.compute(ddata[["ip","app","device","os","channel","click_time"]])[0]
df['day'] = df['click_time'].dt.day.astype(int)
df['hour'] = df['click_time'].dt.hour.astype(int)
df['minute'] = df['click_time'].dt.minute.astype(int)
df['second'] = df['click_time'].dt.second.astype(int)
df.drop(['click_time'], axis='columns', inplace=True)
test_ds = df_to_dataset(df, shuffle=False, batch_size=256, labels=False).prefetch(tf.data.experimental.AUTOTUNE)
predictions=model.predict(test_ds,workers=6,verbose=1)
print("Loading output...")

ddata = dd.read_csv("sample_submission.csv")
df = d.compute(ddata)[0]
df['is_attributed'] = predictions
df.to_csv('sample_submission.csv', index=False)
print("Saved!")
