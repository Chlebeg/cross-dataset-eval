import numpy as np
import pandas as pd
import glob
import os
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import tree
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


AWID2 = "C:/Users/2010m/VSCodeProjects/cross-dataset-eval/datasets/AWID2_downsampled"
AWID3 = "C:/Users/2010m/VSCodeProjects/cross-dataset-eval/datasets/AWID3_downsampled"

train_path = AWID2
test_path = AWID3

awid2_df = pd.read_csv(train_path).sort_index(axis=1)
awid3_df = pd.read_csv(test_path).sort_index(axis=1)

# Concatenate the datasets
both_df = pd.concat([awid2_df, awid3_df], axis=0)

# StratifiedKFold setup for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Separate features and labels
X = both_df.drop(columns=["Label"])
Y = both_df["Label"]

# Encode target labels
le = LabelEncoder()
Y = pd.Series(le.fit_transform(Y))  # Convert back to pandas Series

for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y)):
    print(f"Fold {fold + 1}")

    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = Y.iloc[train_idx], Y.iloc[val_idx]

    # Set up LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Define model parameters
    params = {
        'objective': 'multiclass',  
        'num_classes': 3,
        'metric': 'multi_logloss',
        'boosting': 'gbdt',
        'learning_rate': 0.01,
        'verbose': 1,
        'n_estimators': 80,
        'num_leaves': 80,
        'n_jobs': 4,
        'class_weight': "balanced"
    }

    # Train the model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data]
    )

    # Predict probabilities for validation set
    y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)

    # Get class predictions from probabilities
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate accuracy for this fold
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}\n")
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

    # AUC Score calculation for multiclass
    auc_score = roc_auc_score(y_val, y_pred_proba, multi_class="ovr")
    print(f"AUC Score: {auc_score:.4f}\n")

