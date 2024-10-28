import numpy as np
import pandas as pd
import os
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

USED_FEATURES_AWID3 = [
    "frame.len", "radiotap.length", "radiotap.dbm_antsignal", "wlan.duration",
    "radiotap.present.tsft", "radiotap.channel.freq", "radiotap.channel.flags.cck", "radiotap.channel.flags.ofdm",
    "wlan.fc.type", "wlan.fc.subtype", "wlan.fc.ds", "wlan.fc.frag",
    "wlan.fc.retry", "wlan.fc.pwrmgt", "wlan.fc.moredata", "wlan.fc.protected", "Label"]

USED_FEATURES_AWID3_WO_FREQ = [
    "frame.len", "radiotap.length", "radiotap.dbm_antsignal", "wlan.duration",
    "radiotap.present.tsft", "radiotap.channel.flags.cck", "radiotap.channel.flags.ofdm",
    "wlan.fc.type", "wlan.fc.subtype", "wlan.fc.ds", "wlan.fc.frag",
    "wlan.fc.retry", "wlan.fc.pwrmgt", "wlan.fc.moredata", "wlan.fc.protected", "Label"]

FEATURES_MIN_MAX_SCALING = ["frame.len", "radiotap.length", "radiotap.dbm_antsignal", "wlan.duration"]
# rememebr that here the freq is dropped
FEATURES_ONE_HOT_ENCODING = [
    "radiotap.present.tsft", "radiotap.channel.flags.cck", "radiotap.channel.flags.ofdm",
    "wlan.fc.type", "wlan.fc.subtype", "wlan.fc.ds", "wlan.fc.frag",
    "wlan.fc.retry", "wlan.fc.pwrmgt", "wlan.fc.moredata", "wlan.fc.protected"]

AWID3_DIR = "D:/AWID3/CSV"
AWID3_DIR_PRE = "D:/AWID3/CSV-pre"
AWID3_MERGED = "D:\AWID3\merged_AWID3"
awid3_columns = open("D:/AWID2/code-features/features.txt", "r").read().replace('\n', ' ').split(' ')

### Dtype dict for preprocessing part - not a final one
dtype_dict_AWID3 = {
    "frame.len": "int64", 
    "radiotap.length": "int64",
    "radiotap.dbm_antsignal": "object", 
    "wlan.duration": "int64",
    "radiotap.present.tsft": "object", 
    "radiotap.channel.freq": "int64",
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

def loadDataFrame(file_path):
    df = pd.read_csv(file_path)
    # for x in df.columns:
    #     print(df[x].dtype)
    #     print(f"-----{x}-----\n{df[x].value_counts()}")
    #     print(f"\n")
    return df


def limitFeatures(df):
    # print(f"------1.------\n{df.value_counts}")

    # Change missing values to '-1'
    df = df.replace('?', -1)

    ### Drop nonunique rows - look into it in future
    # df = df[~df.duplicated(keep=False)]
    # print(f"------2.------\n{df.value_counts}")
    
    # Limit to the list of feat.
    df = df[USED_FEATURES_AWID3]
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
            df = loadDataFrame(os.path.join(dir, folder, f))
            print(df["Label"].value_counts())
            df_preporc = limitFeatures(df)

            output_file_path = os.path.join("..", dir+"-pre", f)
            df_preporc.to_csv(output_file_path, index=False)

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
        del df
        gc.collect()

    output_file_path = os.path.join(AWID3_MERGED)
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv(output_file_path, index=False)

    return final_df

def process_antsig(value):
    # Split the string by "-" and convert each part to an integer
    values = [float(v) for v in value.split('-') if v]  # avoid empty splits
    # Return the value itself if only one, otherwise return the average
    return abs(values[0]) if len(values) == 1 else abs(sum(values) / len(values))

def preprocessAWID3(df):
    ### Map wlan.fc.ds
    df["wlan.fc.ds"] = df['wlan.fc.ds'].map(mapping_wlan_fc_ds)

    ### Map radiotap.present.tsft
    df["radiotap.present.tsft"] = df['radiotap.present.tsft'].map(mapping_radiotap_present_tsft)

    ### Map Label
    df["Label"] = df['Label'].map(mapping_label)

    ### Find avg from antsignal (for multiple antenas)
    df["radiotap.dbm_antsignal"] = df["radiotap.dbm_antsignal"].apply(process_antsig)

    ### Dropped freq for now
    df = df[USED_FEATURES_AWID3_WO_FREQ]

    ### Perform min-max scaling
    scaler = MinMaxScaler()
    df[FEATURES_MIN_MAX_SCALING] = scaler.fit_transform(df[FEATURES_MIN_MAX_SCALING])

    ### Perform OneHotEncoding
    df = pd.get_dummies(df, columns=FEATURES_ONE_HOT_ENCODING, drop_first=True)

    ### There're missing columns for ".freq" that needs to be added to match AWID2', 'radiotap.channel.freq
    # all_freq = ['radiotap.channel.freq2412', 'radiotap.channel.freq2417', 'radiotap.channel.freq2422', 'radiotap.channel.freq2427',
    #             'radiotap.channel.freq2432', 'radiotap.channel.freq2437', 'radiotap.channel.freq2442', 'radiotap.channel.freq2447',
    #             'radiotap.channel.freq2452', 'radiotap.channel.freq2457', 'radiotap.channel.freq2462', 'radiotap.channel.freq2467',
    #             'radiotap.channel.freq2472', 'radiotap.channel.freq2484', 'radiotap.channel.freq5180'] 
    # for category in all_freq:
    #     if category not in df.columns:
    #         df[category] = 0
    # ### One column should be dropped, it has to be one of the columns with only zeroes
    
    ### Save dataset to file
    output_file_path = os.path.join(AWID3_DIR, "..", "ready_AWID3")
    print(f"Saving dataset to {output_file_path}")
    df.to_csv(output_file_path, index=False)

    return df

def underSample(df):
    # Separate the classes
    normal_class = df[df['Label'] == 'Normal']
    flooding_class = df[df['Label'] == 'Flooding']
    impersonation_class = df[df['Label'] == 'Impersonation']

    class_to_downsample = normal_class
    # Downsample the normal class x times
    for x in range(8):
        downsampled_class = resample(class_to_downsample, 
                                     replace=False,
                                     n_samples=int(len(class_to_downsample)*0.75),
                                     random_state=42
                                     )
        class_to_downsample = downsampled_class

    # Combine the downsampled normal class with the other classes
    df_balanced = pd.concat([downsampled_class, flooding_class, impersonation_class])

    # Shuffle the resulting dataset (optional)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced dataset class distribution:\n{df_balanced['Label'].value_counts()}")
    output_file_path = os.path.join(AWID3_DIR, "..", "ready_AWID3_downsampled")
    print(f"Saving dataset to {output_file_path}")
    df_balanced.to_csv(output_file_path, index=False)

    

### ---------------- Start ----------------

# df = loadDataFrame("D:/AWID3/CSV/12.Evil_Twin/Evil_Twin_56.csv")
# limitFeatures(df)

cycleThoughFiles(AWID3_DIR)
df = concatFiles(AWID3_DIR_PRE)

df = pd.read_csv(AWID3_MERGED, dtype=dtype_dict_AWID3)
df = preprocessAWID3(df)

df = pd.read_csv(os.path.join(AWID3_DIR, "..", "ready_AWID3"))
underSample(df)

# for x in df.columns:
#     print(df[x].dtype)
#     print(f"-----{x}-----\n{df[x].value_counts()}")
#     print(f"\n")



### ------------ General plan -------------

# frame.len w int
# radiotap.length w int
# radiotap.dbm_antsignal w int, 
    # AWID3 trzeba przerobić potrójne liczby        <-------------- A-3 potencjalnie (avg, mean, first number, min or max)
# wlan.duration w int
# radiotap.present.tsft w int, 
    # AWID3 trzeba przerobić potrójne liczby
# radiotap.channel.freq w int, usunąć wartości -1,
    # trzeba ustawić kolumny [2412,2417,2422,2427,2432,2437,2442, <------- To chyba nie ma sensu
    # 2447,2452,2457,2462,2467,2472,2484,5180]
# radiotap.channel.flag.cck w int
    # AWID2 zmiana nazwy kolumny
# radiotap.channel.flags.ofdm w int
    # AWID2 zmiana nazwy kolumny
# wlan.fc.type w int
# wlan.fc.subtype w int
    # trzeba ustawić kolumny [0-5, 7-15]            
    # w AWID2 brakuje wartości 7, 14 i 15
# wlan.fc.ds
    # 0x00000000 -> 0x00
    # 0x00000001 -> 0x01
    # 0x00000002 -> 0x02
    # 0x00000003 -> 0x03
# wlan.fc.frag w int
# wlan.fc.retry w int
# wlan.fc.pwrmgt w int
# wlan.fc.moredata w int
# wlan.fc.protected w int
# class
    # Z AWID2 usunąć injection
    # W AWID3 zmapować:
        # flooding <- Deauth, Disas, (Re)Assoc, Kr00K
        # impersonation <- RogueAP, Evil_Twin, Krack