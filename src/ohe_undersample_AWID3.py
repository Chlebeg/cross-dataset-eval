import numpy as np
import pandas as pd
import os
from sklearn.utils import resample

USED_FEATURES = [
    "frame.len", "radiotap.length", "radiotap.dbm_antsignal", "wlan.duration",
    "radiotap.present.tsft", "radiotap.channel.flags.cck", "radiotap.channel.flags.ofdm",
    "wlan.fc.type", "wlan.fc.subtype", "wlan.fc.ds", "wlan.fc.frag",
    "wlan.fc.retry", "wlan.fc.pwrmgt", "wlan.fc.moredata", "wlan.fc.protected", "Label"]
FEATURES_ONE_HOT_ENCODING = [
    "radiotap.present.tsft", "radiotap.channel.flags.cck", "radiotap.channel.flags.ofdm",
    "wlan.fc.type", "wlan.fc.subtype", "wlan.fc.ds", "wlan.fc.frag",
    "wlan.fc.retry", "wlan.fc.pwrmgt", "wlan.fc.moredata", "wlan.fc.protected"]

ROOT = "/home/test"
AWID_DIR = ROOT + "/AWID3"
AWID_MERGED = AWID_DIR + "/AWID3_merged"
AWID_SCALED = AWID_DIR + "/AWID3_scaled"
AWID_READY = AWID_DIR + "/AWID3_ready"
AWID_DOWNSAMPLED = AWID_DIR + "/AWID3_downsampled"

def oneHotEncodeAWID3(df):
### Perform OneHotEncoding
    df = pd.get_dummies(df, columns=FEATURES_ONE_HOT_ENCODING, drop_first=True)

    ### Add missing columns for wlan.fc.subtype
    missing_subtypes = ['wlan.fc.type_3', 'wlan.fc.subtype_6']
    for subtype in missing_subtypes:
        df[subtype] = False
    
    ### Save dataset to file
    #output_file_path = os.path.join(AWID_READY)
    #print(f"Saving dataset to {output_file_path}")
    #print(df.columns)
    #df.to_csv(output_file_path, index=False)

    return df

def underSample(df):
    # Separate the classes
    normal_class = df[df['Label'] == 'Normal']
    flooding_class = df[df['Label'] == 'Flooding']
    impersonation_class = df[df['Label'] == 'Impersonation']

    # Downsample to match number of "attack" frames
    downsampled_class = resample(
        normal_class,
        replace=False,
        n_samples=int(sum((flooding_class.shape[0],impersonation_class.shape[0]))/2),
        random_state=42
        )

    # Combine the downsampled normal class with the other classes
    df_balanced = pd.concat([downsampled_class, flooding_class, impersonation_class])
    # Shuffle the resulting dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced dataset class distribution:\n{df_balanced['Label'].value_counts()}")
    output_file_path = os.path.join(AWID_DOWNSAMPLED)
    print(f"Saving dataset to {output_file_path}")
    df_balanced.to_csv(output_file_path, index=False)
    return df_balanced

binary_map = {
    'Normal'        : "Normal",
    'Flooding'      : "Attack",
    'Impersonation' : "Attack",
}

def mapToBinary(df):
    df["Label"] = df['Label'].map(binary_map)
    return df

def undersampleBinary(df, suffix= ""):
    # Separate the classes
    normal_class = df[df['Label'] == "Normal"]
    attack_class = df[df['Label'] == "Attack"]

    # Downsample to match number of "attack" frames
    downsampled_class = resample(
        normal_class,
        replace=False,
        n_samples=int(attack_class.shape[0]),
        random_state=42
        )

    # Combine the downsampled normal class with the other classes
    df_balanced = pd.concat([downsampled_class, attack_class])
    # Shuffle the resulting dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced dataset class distribution:\n{df_balanced['Label'].value_counts()}")
    output_file_path = os.path.join(AWID_DOWNSAMPLED+suffix)
    print(f"Saving dataset to {output_file_path}")
    df_balanced.to_csv(output_file_path, index=False)
    return df_balanced

def dropImpersonationAndUndersample(df, suffix= ""):
    # Separate the classes
    normal_class = df[df['Label'] == "Normal"]
    flooding_class = df[df['Label'] == "Flooding"]

    # Downsample to match number of "attack" frames
    downsampled_class = resample(
        normal_class,
        replace=False,
        n_samples=int(flooding_class.shape[0]),
        random_state=42
        )

    # Combine the downsampled normal class with the other classes
    df_balanced = pd.concat([downsampled_class, flooding_class])
    # Shuffle the resulting dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced dataset class distribution:\n{df_balanced['Label'].value_counts()}")
    output_file_path = os.path.join(AWID_DOWNSAMPLED+suffix)
    print(f"Saving dataset to {output_file_path}")
    df_balanced.to_csv(output_file_path, index=False)
    return df_balanced


### ---------------- Start ----------------

df = pd.read_csv(os.path.join(AWID_SCALED))
df = oneHotEncodeAWID3(df)
# df = pd.read_csv(os.path.join(AWID_READY))
# df = underSample(df)

# df = mapToBinary(df)
# undersampleBinary(df, "_BIN")

dropImpersonationAndUndersample(df, "_NF")


# for x in df.columns:
#     print(df[x].dtype)
#     print(f"-----{x}-----\n{df[x].value_counts()}")
#     print(f"\n")
