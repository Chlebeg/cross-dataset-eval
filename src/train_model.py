import numpy as np
import pandas as pd
import glob
import os
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

AWID2 = "D:/AWID2/ready_AWID2"
AWID3 = "D:/AWID3/ready_AWID3"

def train(df):
    columns = list(df.columns)
    columns.remove("Label")
    trn_X = df[columns]
    trn_Y = df["Label"]
    print("Starting to train")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(trn_X, trn_Y)
    print("Training ended")
    return model

def test(df, model):
    columns = list(df.columns)
    columns.remove("Label")
    tst_X = df[columns]
    tst_Y = df["Label"]
    print("Starting to test")
    pred_Y = model.predict(tst_X)
    print("Testing ended")
    print("Unique values in tst_Y:", set(tst_Y))
    print("Unique values in pred_Y:", set(pred_Y))
    print("Confusion Matrix:") 
    print(confusion_matrix(tst_Y, pred_Y))

    print("\nClassification Report:")
    print(classification_report(tst_Y, pred_Y))


### ----------- Running commands ---------

train_path = AWID3
test_path = AWID2

train_df = pd.read_csv(train_path).sort_index(axis=1)
# print("Train_df loaded")
# model = train(train_df)

test_df = pd.read_csv(test_path).sort_index(axis=1)
# print("Test_df loaded")
# test(test_df, model)

# train_df = pd.read_csv(train_path).sort_index(axis=1)
# test_df = pd.read_csv(test_path).sort_index(axis=1)
print(train_df["Label"].value_counts())
print(test_df["Label"].value_counts())