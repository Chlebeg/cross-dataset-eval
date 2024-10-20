import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

USED_FEATURES = [
    "frame.len", "radiotap.length", "radiotap.dbm_antsignal", "wlan.duration",
    "radiotap.present.tsft", "radiotap.channel.freq", "radiotap.channel.type.cck", "radiotap.channel.type.ofdm",
    "wlan.fc.type", "wlan.fc.subtype", "wlan.fc.ds", "wlan.fc.frag",
    "wlan.fc.retry", "wlan.fc.pwrmgt", "wlan.fc.moredata", "wlan.fc.protected", "class"]
FEATURES_MIN_MAX_SCALING = ["frame.len", "radiotap.length", "radiotap.dbm_antsignal", "wlan.duration"]
FEATURES_ONE_HOT_ENCODING = [
    "radiotap.present.tsft", "radiotap.channel.freq", "radiotap.channel.type.cck", "radiotap.channel.type.ofdm",
    "wlan.fc.type", "wlan.fc.subtype", "wlan.fc.ds", "wlan.fc.frag",
    "wlan.fc.retry", "wlan.fc.pwrmgt", "wlan.fc.moredata", "wlan.fc.protected"]

AWID2_DIR = "D:/AWID2/DATASET_AWID2_CSV/"

# AWID3_DIR = "D:/AWID3"
# merged = "D:\merged_dataset\merged_full.csv"

awid2_columns = open("D:/AWID2/code-features/features.txt", "r").read().replace('\n', ' ').split(' ')

# print(awid2_columns)
# print(len(awid2_columns))

# merged_dataset = pd.read_csv()
# merged_dataset.columns()

# Get AWID2
awid2_trn = pd.read_csv(
    filepath_or_buffer=AWID2_DIR + "AWID-CLS-R-Trn/1",
    # names=awid2_columns
    )
awid2_tst = pd.read_csv(
    filepath_or_buffer=AWID2_DIR + "AWID-CLS-R-Tst/1",
    # names=awid2_columns
    )

awid2_trn.columns = awid2_columns
awid2_tst.columns = awid2_columns

def preprocessAwid2(dataset):
    # print(f"------1.------\n{dataset.value_counts}")

    # Change missing values to '-1'
    dataset = dataset.replace('?', -1)

    # Drop nonunique rows - look into it in future
    # dataset = dataset[~dataset.duplicated(keep=False)]
    # print(f"------2.------\n{dataset.value_counts}")

    # Limit to the list of feat.
    dataset = dataset[USED_FEATURES]
    # print(f"------3.------\n{dataset.value_counts}")

    # min-max scaling
    scaler = MinMaxScaler()
    dataset[FEATURES_MIN_MAX_SCALING] = scaler.fit_transform(dataset[FEATURES_MIN_MAX_SCALING])
    # print(f"------4.------\n{dataset.value_counts}")

    # weird translation - solution for doubled values later in OHE
    dataset["radiotap.channel.type.cck"] = pd.to_numeric(dataset["radiotap.channel.type.cck"], errors='coerce')
    dataset["radiotap.channel.type.ofdm"] = pd.to_numeric(dataset["radiotap.channel.type.ofdm"], errors='coerce')
    dataset["radiotap.channel.freq"] = pd.to_numeric(dataset["radiotap.channel.freq"], errors='coerce')

    # OneHotEncoding
    dataset = pd.get_dummies(dataset, columns=FEATURES_ONE_HOT_ENCODING, drop_first=True)
    # print(f"------5.------\n{dataset.columns}")
    # print(len(dataset.columns))
    # print(dataset.head())

    # for x in dataset.columns:
    #     print(dataset[x].dtype)
    #     print(f"-----{x}-----\n{dataset[x].value_counts()}")
    #     print(f"\n")
    return dataset

ready_trn = preprocessAwid2(awid2_trn)
ready_tst = preprocessAwid2(awid2_tst)

# trn_col = ready_trn.columns.to_list()
# tst_col = ready_tst.columns.to_list()
# print(trn_col)
# print(tst_col)

col = ['frame.len', 'radiotap.length', 'radiotap.dbm_antsignal', 'wlan.duration', 'radiotap.present.tsft_1', 'radiotap.channel.freq_2412', 'radiotap.channel.freq_2417', 'radiotap.channel.freq_2422', 'radiotap.channel.freq_2427', 'radiotap.channel.freq_2432', 'radiotap.channel.freq_2437', 'radiotap.channel.freq_2442', 'radiotap.channel.freq_2447', 'radiotap.channel.freq_2452', 'radiotap.channel.freq_2457', 'radiotap.channel.freq_2462', 'radiotap.channel.freq_2467', 'radiotap.channel.freq_2472', 'radiotap.channel.freq_2484', 'radiotap.channel.type.cck_0', 'radiotap.channel.type.cck_1', 'radiotap.channel.type.ofdm_0', 'radiotap.channel.type.ofdm_1', 'wlan.fc.type_1', 'wlan.fc.type_2', 'wlan.fc.subtype_1', 'wlan.fc.subtype_2', 'wlan.fc.subtype_3', 'wlan.fc.subtype_4', 'wlan.fc.subtype_5', 'wlan.fc.subtype_8', 'wlan.fc.subtype_10', 'wlan.fc.subtype_11', 'wlan.fc.subtype_12', 'wlan.fc.subtype_13', 'wlan.fc.ds_0x01', 'wlan.fc.ds_0x02', 'wlan.fc.frag_1', 'wlan.fc.retry_1', 'wlan.fc.pwrmgt_1', 'wlan.fc.moredata_1', 'wlan.fc.protected_1']

trn_X = ready_trn[col]
trn_Y = ready_trn["class"]
tst_X = ready_tst[col]
tst_Y = ready_tst["class"]

#training section
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(trn_X, trn_Y)
pred_Y = rf_model.predict(tst_X)

print("Confusion Matrix:")
print(confusion_matrix(tst_Y, pred_Y))

print("\nClassification Report:")
print(classification_report(tst_Y, pred_Y))

# Preprocess AWID2
# Preprocess AWID3
# Train AWID2 on AWID3 

# Notes:
# - probably a good idea to delete non-unique before reducing the size of features
# - probably a good idea to drop all the lines with -1 values (for reduced dataset it was around 780 entries), investigate later
# - 
