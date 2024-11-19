import numpy as np
import pandas as pd
import os

USED_FEATURES = [
    "frame.len", "radiotap.length", "radiotap.dbm_antsignal", "wlan.duration",
    "radiotap.present.tsft", "radiotap.channel.flags.cck", "radiotap.channel.flags.ofdm",
    "wlan.fc.type", "wlan.fc.subtype", "wlan.fc.ds", "wlan.fc.frag",
    "wlan.fc.retry", "wlan.fc.pwrmgt", "wlan.fc.moredata", "wlan.fc.protected", "Label"]

ROOT = "/root/learning"
AWID_DIR = ROOT + "/AWID3"
AWID_CSV = AWID_DIR + "/CSV"
AWID_CSV_PRE = AWID_DIR + "/CSV-pre"
AWID_MERGED = AWID_DIR + "/AWID3_merged"

### Dtype dict for preprocessing part - not a final one
dtype_dict_AWID3 = {
    "frame.len": "float64", 
    "radiotap.length": "float64",
    "radiotap.dbm_antsignal": "object", 
    "wlan.duration": "float64",
    "radiotap.present.tsft": "object", 
    # "radiotap.channel.freq": "int64",
    "radiotap.channel.flags.cck": "int64", 
    "radiotap.channel.flags.ofdm": "int64",
    "wlan.fc.type": "int64", 
    "wlan.fc.subtype": "int64", 
    "wlan.fc.ds": "object", 
    "wlan.fc.frag": "int64",
    "wlan.fc.retry": "int64", 
    "wlan.fc.pwrmgt": "int64", 
    "wlan.fc.moredata": "int64", 
    "wlan.fc.protected": "int64", 
    "Label": "category"
}
mapping_wlan_fc_ds = {
    '0x00000000': 0,
    '0x00000001': 1,
    '0x00000002': 2,
    '0x00000003': 3
}
mapping_radiotap_present_tsft = {
    '0-0-0': 0,
    '1-0-0': 1,
}
mapping_label = {
    'Normal'    : 'Normal',
    'Deauth'    : 'Flooding',
    'Disas'     : 'Flooding',
    '(Re)Assoc' : 'Flooding',
    'Kr00K'     : 'Flooding',
    'Kr00k'     : 'Flooding',
    'RogueAP'   : 'Impersonation',
    'Evil_Twin' : 'Impersonation',
    'Krack'     : 'Impersonation'
}


def limitFeatures(df):
    ### Replace missing numbers with NaN
    df = df.replace('?', np.nan)

    ### Drop nonunique rows - look into it in future
    # df = df[~df.duplicated(keep=False)]
    # print(f"------2.------\n{df.value_counts}")
    
    # Limit to the list of feat.
    df = df[USED_FEATURES]
    # Drop all that are still invalid
    df = df.dropna()
    
    # Change types of rows to correct ones
    df = df.astype(dtype_dict_AWID3)
    return df

def cycleThoughFiles(dir):
    folders = os.listdir(dir)
    for folder in folders:
        files = os.listdir(os.path.join(dir, folder))
        print(f"Files to cycle: {files}")
        for f in files:
            print(f"Right now going through {f}")
            df = pd.read_csv(os.path.join(dir, folder, f))
            df_preporc = limitFeatures(df)

            output_file_path = os.path.join(AWID_CSV_PRE, f)
            df_preporc.to_csv(output_file_path, index=False)

### Unused function for listing sets of values in datasets
# def cycleThoughFilesAndListFeatVal(dir):
#     try:
#         folders = os.listdir(dir)
#         dict = {}
#         for folder in folders:
#             files = os.listdir(os.path.join(dir, folder))
#             print(f"Files to cycle: {files}")
#             for f in files:
#                 print(f"Right now going through {f}")
#                 df = loadDataFrame(os.path.join(dir, folder, f))
#                 df = limitFeatures(df)

#                 for x in FEATURES_ONE_HOT_ENCODING:
#                     if x not in dict.keys():
#                         dict[x] = []
#                     for val in set(df[x]):
#                         if val not in dict[x]:
#                             dict[x].append(val)
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
    output_file_path = os.path.join(AWID_MERGED)
    print(f"Saving merged dataset to {output_file_path}")
    final_df.to_csv(output_file_path, index=False)
    return final_df

def process_antsig(value):
    # Split the string by "-" and convert each part to an integer
    values = [float(v) for v in value.split('-') if v]  # avoid empty splits
    # Return the value itself if only one, otherwise return the average
    return -values[0] if len(values) == 1 else -float(round(sum(values) / len(values)))

def preprocessAWID3(df):
    ### Map wlan.fc.ds
    df["wlan.fc.ds"] = df['wlan.fc.ds'].map(mapping_wlan_fc_ds)

    ### Map radiotap.present.tsft
    df["radiotap.present.tsft"] = df['radiotap.present.tsft'].map(mapping_radiotap_present_tsft)

    ### Map Label
    df["Label"] = df['Label'].map(mapping_label)

    ### Find avg from antsignal (for multiple antenas)
    df["radiotap.dbm_antsignal"] = df["radiotap.dbm_antsignal"].apply(process_antsig)

    ### Save dataset to file
    output_file_path = os.path.join(AWID_MERGED)
    print(f"Saving dataset to {output_file_path}")
    df.to_csv(output_file_path, index=False)
    return df



### ---------------- Start ----------------

cycleThoughFiles(AWID_CSV)
df = concatFiles(AWID_CSV_PRE)

df = pd.read_csv(AWID_MERGED, dtype=dtype_dict_AWID3)
df = preprocessAWID3(df)

# for x in df.columns:
#     print(df[x].dtype)
#     print(f"-----{x}-----\n{df[x].value_counts()}")
#     print(f"\n")
