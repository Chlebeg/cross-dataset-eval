import numpy as np
import pandas as pd
import os
import gc
from sklearn.preprocessing import MinMaxScaler

USED_FEATURES = [
    "frame.len", "radiotap.length", "radiotap.dbm_antsignal", "wlan.duration",
    "radiotap.present.tsft", "radiotap.channel.freq", "radiotap.channel.type.cck", "radiotap.channel.type.ofdm",
    "wlan.fc.type", "wlan.fc.subtype", "wlan.fc.ds", "wlan.fc.frag",
    "wlan.fc.retry", "wlan.fc.pwrmgt", "wlan.fc.moredata", "wlan.fc.protected", "class"]

USED_FEATURES_WO_ANTSIGNAL_AND_FREQ = [
    "frame.len", "radiotap.length", "wlan.duration",
    "radiotap.present.tsft", "radiotap.channel.type.cck", "radiotap.channel.type.ofdm",
    "wlan.fc.type", "wlan.fc.subtype", "wlan.fc.ds", "wlan.fc.frag",
    "wlan.fc.retry", "wlan.fc.pwrmgt", "wlan.fc.moredata", "wlan.fc.protected", "class"]

# rememebr that here the antsingal is dropped
FEATURES_MIN_MAX_SCALING = ["frame.len", "radiotap.length", "wlan.duration"]
# rememebr that here the freq is dropped
FEATURES_ONE_HOT_ENCODING = [
    "radiotap.present.tsft", "radiotap.channel.type.cck", "radiotap.channel.type.ofdm",
    "wlan.fc.type", "wlan.fc.subtype", "wlan.fc.ds", "wlan.fc.frag",
    "wlan.fc.retry", "wlan.fc.pwrmgt", "wlan.fc.moredata", "wlan.fc.protected"]

AWID2_DIR = "D:/AWID2/DATASET_AWID2_CSV/"
AWID2_DIR_PRE = "D:/AWID2/DATASET_AWID2_CSV/"
TRN_FOLDER = AWID2_DIR + "AWID-CLS-R-Trn"
TST_FOLDER = AWID2_DIR + "AWID-CLS-R-Tst"
awid2_columns = open("D:/AWID2/code-features/features.txt", "r").read().replace('\n', ' ').split(' ')

### Dtype dict for preprocessing part - not a final one
dtype_dict_AWID2 = {
    "frame.len": "int64", 
    "radiotap.length": "int64",
    "radiotap.dbm_antsignal": "object", 
    "wlan.duration": "int64",
    "radiotap.present.tsft": "object", 
    "radiotap.channel.freq": "int64",
    "radiotap.channel.type.cck": "int64", 
    "radiotap.channel.type.ofdm": "int64",
    "wlan.fc.type": "int64", 
    "wlan.fc.subtype": "int64", 
    "wlan.fc.ds": "object", 
    "wlan.fc.frag": "int64",
    "wlan.fc.retry": "int64", 
    "wlan.fc.pwrmgt": "int64", 
    "wlan.fc.moredata": "int64", 
    "wlan.fc.protected": "int64", 
    "class": "category"
}

mapping_wlan_fc_ds = {
    '0x00': 0,
    '0x01': 1,
    '0x02': 2,
}

mapping_class = {
    'normal'        : 'Normal',
    'flooding'      : 'Flooding',
    'impersonation' : 'Impersonation',
}


def loadDataFrame(file_path):
    df = pd.read_csv(file_path)
    df.columns = awid2_columns
    # for x in df.columns:
    #     print(df[x].dtype)
    #     print(f"-----{x}-----\n{df[x].value_counts()}")
    #     print(f"\n")
    return df

def limitFeatures(df):
    # print(f"------1.------\n{df.value_counts}")

    ### Change missing values to '-1'
    df = df.replace('?', -1)

    ### Drop nonunique rows - look into it in future
    # df = df[~df.duplicated(keep=False)]
    # print(f"------2.------\n{df.value_counts}")
    
    ### Limit to the list of feat.
    df = df[USED_FEATURES]
    ### Drop all that are still invalid
    df = df.dropna()
    
    ### Change types of rows to correct ones
    df = df.astype(dtype_dict_AWID2)
    return df
    

def preprocessAWID2(df):
    ### Drop all frames with 'injection' class
    df = df[df['class'] != 'injection']
    
    ### Drop all corrupted frames
    df = df[df['radiotap.channel.type.ofdm'] != -1]

    ### Map wlan.fc.ds
    df["wlan.fc.ds"] = df['wlan.fc.ds'].map(mapping_wlan_fc_ds)

    ### Rename class
    df["class"] = df['class'].map(mapping_class)

    ### For now not sure what to do with antsignal so I'll drop this one
    df = df[USED_FEATURES_WO_ANTSIGNAL_AND_FREQ]

    ### Perform min-max scaling
    scaler = MinMaxScaler()
    df[FEATURES_MIN_MAX_SCALING] = scaler.fit_transform(df[FEATURES_MIN_MAX_SCALING])

    ### Perform OneHotEncoding
    df = pd.get_dummies(df, columns=FEATURES_ONE_HOT_ENCODING, drop_first=True)

    ### Add missing columns for wlan.fc.subtype
    missing_subtypes = ['wlan.fc.subtype_7', 'wlan.fc.subtype_14', 'wlan.fc.subtype_15']
    for subtype in missing_subtypes:
        df[subtype] = 0

    ### Add missing column for wlan.fc.ds
    df['wlan.fc.ds_3'] = 0

    ### Add missing columns
    df['radiotap.present.tsft_1'] = 0
    df["wlan.fc.frag_1"] = 0

    ### Change "radiotap.channel.type.cck", "radiotap.channel.type.ofdm" and "class" column names to flags to match AWID3
    df.rename(columns = {'radiotap.channel.type.cck_1':'radiotap.channel.flags.cck_1',
                        'radiotap.channel.type.ofdm_1': 'radiotap.channel.flags.ofdm_1',
                        'class': 'Label'}, inplace = True)
    
    ### Save dataset to file
    output_file_path = os.path.join(AWID2_DIR, "..", "ready_AWID2")
    print(f"Saving dataset to {output_file_path}")
    df.to_csv(output_file_path, index=False)

    return df


### ---------------- Start ----------------

df = loadDataFrame("D:/AWID2/DATASET_AWID2_CSV/AWID-CLS-R-Trn/1")
df = limitFeatures(df)

df = preprocessAWID2(df)

for x in df.columns:
    print(df[x].dtype)
    print(f"-----{x}-----\n{df[x].value_counts()}")
    print(f"\n")




### ------------- FOR FULL AWID2 DATASETS ------------- work in progress

# def cycleThoughFiles(folder):
#     files = os.listdir(folder)
#     print(f"Files to cycle: {files}")
#     for f in files:
#         print(f"Right now going through {f}")
#         df = pd.read_csv(os.path.join(folder, f))
#         df.columns = awid2_columns
#         print(df["class"].value_counts())
#         df_preporc = preprocesDataframe(df)

#         output_file_path = os.path.join("..", folder+"-pre", f)
#         df_preporc.to_csv(output_file_path, index=False)
#         gc.collect()

# cycleThoughFiles(TST_FOLDER)