import numpy as np
import pandas as pd
import os

USED_FEATURES_AWID2 = [
    "frame.len", "radiotap.length", "radiotap.dbm_antsignal", "wlan.duration",
    "radiotap.present.tsft", "radiotap.channel.type.cck", "radiotap.channel.type.ofdm",
    "wlan.fc.type", "wlan.fc.subtype", "wlan.fc.ds", "wlan.fc.frag",
    "wlan.fc.retry", "wlan.fc.pwrmgt", "wlan.fc.moredata", "wlan.fc.protected", "class"]

ROOT = "/home/test"
AWID_DIR = ROOT + "/AWID2"
AWID_CSV = AWID_DIR + "/CSV"
AWID_CSV_PRE = AWID_DIR + "/CSV-pre"
AWID_MERGED = AWID_DIR + "/AWID2_merged"
TRN_FOLDER = AWID_CSV + "/AWID-CLS-F-Trn"
TST_FOLDER = AWID_CSV + "/AWID-CLS-F-Tst"

awid2_columns = open(os.path.join(AWID_DIR, "code-features/features.txt"), "r").read().replace('\n', ' ').split(' ')

### Dtype dict for preprocessing part
dtype_dict_AWID2 = {
    "frame.len": "float64", 
    "radiotap.length": "float64",
    "radiotap.dbm_antsignal": "object", 
    "wlan.duration": "float64",
    "radiotap.present.tsft": "int64", 
    # "radiotap.channel.freq": "int64",
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
    '0x03': 3,
}
mapping_class = {
    'normal'        : 'Normal',
    'flooding'      : 'Flooding',
    'impersonation' : 'Impersonation',
}

def limitFeatures(df):
    ### Replace missing numbers with NaN
    df = df.replace('?', np.nan)

    ### Drop nonunique rows - look into it in future
    # df = df[~df.duplicated(keep=False)]
    # print(f"------2.------\n{df.value_counts}")
    
    ### Limit to the list of feat.
    df = df[USED_FEATURES_AWID2]
    ### Drop all rows with NaN values
    df = df.dropna()
    
    ### Change types of rows to correct ones
    df = df.astype(dtype_dict_AWID2)
    return df

def cycleThoughFiles(dir, prefix=""):
    files = os.listdir(dir)
    print(f"Files to cycle: {files}")
    for f in files:
        print(f"Right now going through {f}")
        df = pd.read_csv(os.path.join(dir, f))
        df.columns = awid2_columns
        df_preporc = limitFeatures(df)

        output_file_path = os.path.join(AWID_CSV_PRE, prefix + f)
        df_preporc.to_csv(output_file_path, index=False)

### Unused function for listing sets of values in datasets
# def cycleThoughFilesAndListFeatVal(dir):
#     try:
#         files = os.listdir(dir)
#         print(f"Files to cycle: {files}")
#         dict = {}
#         for f in files:
#             print(f"Right now going through {f}")
#             df = loadDataFrame(os.path.join(dir, f))
#             df = limitFeatures(df)

#             for x in FEATURES_ONE_HOT_ENCODING:
#                 if x not in dict.keys():
#                     dict[x] = []
#                 for val in set(df[x]):
#                     if val not in dict[x]:
#                         dict[x].append(val)
#         print(dict)
#     except:
#         print(dict)

def concatFiles(dir):
    files = os.listdir(os.path.join(dir))
    dfs = []
    for f in files:
        print(f"Right now going through {f}")
        df = pd.read_csv(os.path.join(dir, f))
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)
    return final_df

def process_antsig(value):
    return float(value)

def preprocessAWID2(df):
    ### Drop all frames with 'injection' class
    df = df[df['class'] != 'injection']

    ### Map wlan.fc.ds
    df["wlan.fc.ds"] = df['wlan.fc.ds'].map(mapping_wlan_fc_ds)
    
    ### Rename class
    df["class"] = df['class'].map(mapping_class)

    df["radiotap.dbm_antsignal"] = df["radiotap.dbm_antsignal"].apply(process_antsig)

    ### Change "radiotap.channel.type.cck", "radiotap.channel.type.ofdm" and "class" column names to flags to match AWID3
    df.rename(columns = {'radiotap.channel.type.cck':'radiotap.channel.flags.cck',
                        'radiotap.channel.type.ofdm': 'radiotap.channel.flags.ofdm',
                        'class': 'Label'}, inplace = True)
    
    ### Save dataset to file
    output_file_path = os.path.join(AWID_MERGED)
    print(f"Saving preprocessed dataset to {output_file_path}")
    df.to_csv(output_file_path, index=False)

    return df



### ---------------- Start ----------------

cycleThoughFiles(TRN_FOLDER)
df = concatFiles(AWID_CSV_PRE)
#df = pd.read_csv(AWID_MERGED)
df = preprocessAWID2(df)

# for x in df.columns:
#     print(df[x].dtype)
#     print(f"-----{x}-----\n{df[x].value_counts()}")
#     print(f"\n")
