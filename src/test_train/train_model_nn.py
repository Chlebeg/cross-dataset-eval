import numpy as np
import pandas as pd
import glob
import os
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import tree
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder

AWID2 = "/root/learning/AWID2/AWID2_downsampled"
AWID3 = "/root/learning/AWID3/AWID3_downsampled"
AWID2_tst = "/root/learning/AWID2/ready_AWID2_downsampled_tst"

AWID2_MERGED = "/root/learning/AWID2/merged_AWID2"
AWID3_MERGED = "/root/learning/AWID3/merged_AWID3"

def train(df):
    columns = list(df.columns)
    columns.remove("Label")
    trn_X = df[columns]
    trn_Y = df["Label"]
    
    le = LabelEncoder()
    le.fit(trn_Y)
    trn_Y = le.transform(trn_Y)

    trn_X = np.array(trn_X).astype('float32')
    trn_Y = np.array(trn_Y).astype('float32')
    
    print(trn_X.shape)
    print(trn_Y.shape)
    trn_X = tf.reshape(trn_X, (trn_X.shape[0], trn_X.shape[1], 1))
    trn_Y = tf.keras.utils.to_categorical(trn_Y, len(set(trn_Y)))
   
    print(trn_X.shape)
    print(trn_Y.shape)

    # Define the model
    model = Sequential()

    # Input layer is implicit in the input shape of the first Conv1D layer
    # First Conv1D Layer
    model.add(Conv1D(filters=128, kernel_size=1, strides=1, activation='relu', padding='same', input_shape=(trn_X.shape[1], trn_X.shape[2])))
    model.add(Dropout(0.5))

    # Second Conv1D Layer
    model.add(Conv1D(filters=64, kernel_size=1, strides=1, activation='relu', padding='same'))
    model.add(Dropout(0.5))

    # Third Conv1D Layer
    model.add(Conv1D(filters=32, kernel_size=1, strides=1, activation='relu', padding='same'))
    model.add(Dropout(0.5))

    # Flatten Layer
    model.add(Flatten())

    # Dense Layer with L2 Regularization
    model.add(Dense(units=100, activation='relu', kernel_regularizer=l2(0.1)))

    # Output Layer
    model.add(Dense(3, activation='softmax'))

    #Compile the model (using categorical_crossentropy for multi-class classification)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

    model.fit(trn_X, trn_Y, batch_size=128)
    print("Training ended")
    return model

def test(df, model):
    columns = list(df.columns)
    columns.remove("Label")
    tst_X = df[columns]
    tst_Y = df["Label"]

    le = LabelEncoder()
    le.fit(tst_Y)
    tst_Y = le.transform(tst_Y)

    tst_X = np.array(tst_X).astype('float32')
    tst_Y = np.array(tst_Y).astype('float32')

    tst_X = tf.reshape(tst_X, (tst_X.shape[0], tst_X.shape[1], 1))
    tst_Y = tf.keras.utils.to_categorical(tst_Y, len(set(tst_Y)))

    print("Starting to test")
    pr = model.predict(tst_X)
    print("Testing ended")
    
    pred_y = np.argmax(pr, axis=1)
    tst_Y = np.argmax(tst_Y, axis=1)

    res = classification_report(tst_Y, pred_y, digits=3)
    print(res)

    conf_matrix = confusion_matrix(tst_Y, pred_y)
    print(conf_matrix)

### ----------- Running commands -----------

train_path = AWID2
test_path = AWID3

train_df = pd.read_csv(train_path).sort_index(axis=1)
print("------------- Train_df loaded -------------")
# print(train_df["Label"].value_counts())
model = train(train_df)

test_df = pd.read_csv(test_path).sort_index(axis=1)
print("------------- Test_df loaded --------------")
# print(test_df["Label"].value_counts())
test(test_df, model)

# train_df = pd.read_csv(train_path).sort_index(axis=1)
# test_df = pd.read_csv(test_path).sort_index(axis=1)
# print(train_df["Label"].value_counts())
# print(test_df["Label"].value_counts())
# print(train_df.columns)
# print(test_df.columns)

#for x in train_df.columns:
#    print(train_df[x].dtype)
#    print(f"-----{x}-----\n{train_df[x].value_counts()}")
#    print(f"\n")
#
#for x in test_df.columns:
#    print(test_df[x].dtype)
#    print(f"-----{x}-----\n{test_df[x].value_counts()}")
#    print(f"\n")
