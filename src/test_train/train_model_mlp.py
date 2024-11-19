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

AWID2 = "/root/learning/AWID2/AWID2_downsampled"
AWID3 = "/root/learning/AWID3/AWID3_downsampled"
AWID2_tst = "/root/learning/AWID2/ready_AWID2_downsampled_tst"

AWID2_MERGED = "/root/learning/AWID2/AWID2_merged"
AWID3_MERGED = "/root/learning/AWID3/AWID3_merged"

def train(df):
    columns = list(df.columns)
    columns.remove("Label")
    trn_X = df[columns]
    trn_Y = df["Label"]
    print("Starting to train")
    model = MLPClassifier(
        solver="adam",
        hidden_layer_sizes=(30, 20, 16, 12, 6),
        max_iter=300,
        random_state=42,
        batch_size=200,
        verbose=True
    )
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

    print("Confusion Matrix:") 
    print(confusion_matrix(tst_Y, pred_Y))

    print("\nClassification Report:")
    print(classification_report(tst_Y, pred_Y))

    # y_pred_proba = model.predict_proba(tst_X)
    # auc_score = roc_auc_score(tst_Y, y_pred_proba[:, 1], multi_class='ovr', average="micro")
    # print(f"AUC Score: {auc_score:.4f}")

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

#for x in test_df.columns:
#    print(test_df[x].dtype)
#    print(f"-----{x}-----\n{test_df[x].value_counts()}")
#    print(f"\n")
