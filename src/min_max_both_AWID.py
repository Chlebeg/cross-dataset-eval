import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURES_MIN_MAX_SCALING = ["frame.len", "radiotap.length", "radiotap.dbm_antsignal", "wlan.duration"]

ROOT = "D:/"
AWID2_MERGED = ROOT + "/AWID2/AWID2_merged"
AWID3_MERGED = ROOT + "/AWID3/AWID3_merged"
AWID2_SCALED = ROOT + "/AWID2/AWID2_scaled"
AWID3_SCALED = ROOT + "/AWID3/AWID3_scaled"


def doMinMaxScaling(df1, df2):
    df1_scaled = df1.copy()
    df2_scaled = df2.copy()

    for feat in FEATURES_MIN_MAX_SCALING:
        min_val = min(df1[feat].min(), df2[feat].min())
        max_val = max(df1[feat].max(), df2[feat].max())

        scaler = MinMaxScaler()
        scaler.fit([[min_val], [max_val]])
        df1_scaled[feat] = scaler.transform(df1[[feat]])
        df2_scaled[feat] = scaler.transform(df2[[feat]])

    return df1_scaled, df2_scaled



### ---------------- Start ----------------

awid2 = pd.read_csv(AWID2_MERGED)
awid3 = pd.read_csv(AWID3_MERGED)

awid2_scaled, awid3_scaled = doMinMaxScaling(awid2, awid3)

print(f"Saving dataset to {AWID2_SCALED}")
awid2_scaled.to_csv(AWID2_SCALED, index=False)
print(f"Saving dataset to {AWID3_SCALED}")
awid3_scaled.to_csv(AWID3_SCALED, index=False)

