import argparse
import json
import sys
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.python.keras.backend import clear_session
from keras.src.layers import Conv1D, Dense, Dropout, Flatten, Input, BatchNormalization
from keras.src.models import Sequential
from keras.src.regularizers import L2
from keras.src.optimizers import SGD
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from scipy.special import softmax

from keras.src.callbacks import EarlyStopping

ROOT = "/home/test"
AWID2 = ROOT + "/AWID2/AWID2_downsampled"
AWID3 = ROOT + "/AWID3/AWID3_downsampled"
AWID2_BIN = ROOT + "/AWID2/AWID2_downsampled_BIN" # path to binary AWIDs (Normal/Attack)
AWID3_BIN = ROOT + "/AWID3/AWID3_downsampled_BIN"
AWID2_NF = ROOT + "/AWID2/AWID2_downsampled_NF" # path to [Normal, Flooding] AWIDs dataset
AWID3_NF = ROOT + "/AWID3/AWID3_downsampled_NF"

CONFIG_PATH = ROOT + "/config"
HEATMAPS_PATH = ROOT + "/heatmaps/"

start_time = 0; end_time = 0

def decideOnPath(name):
    match name:
        case "AWID2":
            return AWID2
        case "AWID2_BIN":
            return AWID2_BIN
        case "AWID2_NF":
            return AWID2_NF
        case "AWID3":
            return AWID3
        case "AWID3_BIN":
            return AWID3_BIN
        case "AWID3_NF":
            return AWID3_NF
        case _:
            raise FileNotFoundError(f"Dataset '{name}' not found")

def loadDataset(path):
    print(f"----- {path}... ", end="")
    df = pd.read_csv(path).sort_index(axis=1)
    print("loaded -----")
    return df

def getParamsFromConfig(model_name):
    config_file = f"{CONFIG_PATH}/{model_name}_config.json"
    try:
        with open(config_file, "r") as file:
            params = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file for '{model_name}' not found at {config_file}.")
    return params

def getModel(model_name):
    # 1-Dim Convolutional Neural Network
    def get1DCNN(trn_X_shape, trn_Y_shape):
        model = Sequential()
        model.add(Input(shape=(trn_X_shape[1], trn_X_shape[2])))        
        model.add(Conv1D(filters=128, kernel_size=1, strides=1, activation='relu', padding='same'))
        model.add(Dropout(0.5))
        model.add(Conv1D(filters=64, kernel_size=1, strides=1, activation='relu', padding='same'))
        model.add(Dropout(0.5))
        model.add(Conv1D(filters=32, kernel_size=1, strides=1, activation='relu', padding='same'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(units=100, activation='relu', kernel_regularizer=L2(0.1)))
        model.add(Dense(trn_Y_shape[1], activation='softmax'))
        model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
        return model

    # MLP-name
    def getMLP(trn_X_shape, trn_Y_shape):
        model = Sequential()
        model.add(Input(shape=(trn_X_shape[1],))) 
        model.add(Dense(30, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Dense(20, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(12, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(6, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(trn_Y_shape[1], activation='softmax'))
        model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
                  loss="categorical_crossentropy",
                  metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
        return model

    match model_name:
        # Stochastic-based
        case "lr":
            model = LogisticRegression()
            model.set_params(**getParamsFromConfig(model_name))
            return model
        case "sgdc":
            model = SGDClassifier()
            model.set_params(**getParamsFromConfig(model_name))
            return model
        # Linear-based
        case "lsvc":
            model = LinearSVC()
            model.set_params(**getParamsFromConfig(model_name))
            return model
        # Tree-based
        case "dt":
            model = DecisionTreeClassifier()
            model.set_params(**getParamsFromConfig(model_name))
            return model
        case "rf":
            model = RandomForestClassifier()
            model.set_params(**getParamsFromConfig(model_name))
            return model
        case "et":
            model = ExtraTreesClassifier()
            model.set_params(**getParamsFromConfig(model_name))
            return model
        case "ab":
            model = AdaBoostClassifier()
            model.set_params(**getParamsFromConfig(model_name))
            return model
        case "lgbm":
            model = lgb.LGBMClassifier()
            model.set_params(**getParamsFromConfig(model_name))
            if not multiclass: model.set_params(**{
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "num_classes": 1,
                    "is_unbalance": "False",
                    })
            return model
        # Neural Networks
        case "mlp":
            return getMLP
        case "1dcnn":
            return get1DCNN
        case _:
            raise ValueError(f"Model '{model_name}' is not supported!")

def train(df, model, le):
    columns = list(df.columns)
    columns.remove("Label")
    trn_X = df[columns]
    # trn_Y = df["Label"]
    
    # trn_Y = pd.Series(le.transform(trn_Y))

    trn_Y = df["Label"].to_numpy().reshape(-1, 1)
    trn_Y = le.transform(trn_Y)
    trn_Y = pd.Series(trn_Y.flatten(), name="Label")

    global start_time

    if args.model_name not in ["1dcnn", "sae", "mlp"]:
        # Shallow learning path
        start_time = time.time()
        model.fit(trn_X, trn_Y)
    else: 
        # MLP / 1DCNN path - we change to tensors
        trn_X = np.array(trn_X).astype('float32')
        trn_Y = np.array(trn_Y).astype('float32')
        trn_X = tf.reshape(trn_X, (trn_X.shape[0], trn_X.shape[1], 1))
        trn_Y = tf.keras.utils.to_categorical(trn_Y, len(set(trn_Y)))
        print(trn_X.shape, trn_Y.shape)
        model = model(trn_X.shape, trn_Y.shape)
        early_stopping = EarlyStopping(patience=2)
        start_time = time.time()
        model.fit(trn_X, trn_Y, batch_size=128, validation_split=0.2, epochs=50, callbacks=[early_stopping])
    print("Training completed.")
    return model

def calculateScoreMulticlass(tst_Y, pred_Y_proba):
    pred_Y = np.argmax(pred_Y_proba, axis=1)

    auc_score = roc_auc_score(tst_Y, pred_Y_proba, average="weighted", multi_class="ovo")
    precision = precision_score(tst_Y, pred_Y, average="macro")
    recall = recall_score(tst_Y, pred_Y, average="macro")
    f1 = f1_score(tst_Y, pred_Y, average="macro")
    accuracy = accuracy_score(tst_Y, pred_Y)
    conf_matrix = confusion_matrix(tst_Y, pred_Y)
    return auc_score, precision, recall, f1, accuracy, conf_matrix

def calculateScoreBinary(tst_Y, pred_Y_proba):
    positive_class_prob = pred_Y_proba[:, 1]
    pred_Y = (positive_class_prob >= 0.5).astype(int)

    auc_score = roc_auc_score(tst_Y, positive_class_prob)
    precision = precision_score(tst_Y, pred_Y)
    recall = recall_score(tst_Y, pred_Y)
    f1 = f1_score(tst_Y, pred_Y)
    accuracy = accuracy_score(tst_Y, pred_Y)
    conf_matrix = confusion_matrix(tst_Y, pred_Y)
    return auc_score, precision, recall, f1, accuracy, conf_matrix

def chooseName(model_name):
    match model_name:
        case "lr":
            return "Logistic Regression"
        case "sgdc":
            return "SGDClassifier"
        case "lsvc":
            return "LinearSVC"
        case "dt":
            return "DT"
        case "rf":
            return "RF"
        case "et":
            return "ET"
        case "ab":
            return "AdaBoost"
        case "lgbm":
            return "LightGBM"
        case "mlp":
            return "MLP"
        case "1dcnn":
            return "1DCNN"
        case _:
            raise ValueError(f"Model '{model_name}' is not supported!")

def test(df, model, le):
    columns = list(df.columns)
    columns.remove("Label")
    tst_X = df[columns]
    # tst_Y = df["Label"]

    # tst_Y = pd.Series(le.transform(tst_Y))
    # labels = le.classes_

    labels = le.categories[0]

    tst_Y = df["Label"].to_numpy().reshape(-1, 1)
    tst_Y = le.transform(tst_Y)
    tst_Y = pd.Series(tst_Y.flatten(), name="Label")

    global end_time

    if args.model_name not in ["1dcnn", "sae", "mlp"]:
        # Shallow learning path
        if args.model_name in ["lsvc"]: # lsvc has no pred proba impl
            predict_proba_dist = model.decision_function(tst_X)
            if multiclass:
                pred_Y_proba = softmax(predict_proba_dist, axis=1)
            else:
                pred_Y_proba = softmax(np.column_stack((-predict_proba_dist, predict_proba_dist)), axis=1)
        else:
            pred_Y_proba = model.predict_proba(tst_X)
    else:
        # MLP / 1DCNN path - we change to tensors
        tst_X = np.array(tst_X).astype('float32')
        tst_Y = np.array(tst_Y).astype('float32')    
        tst_X = tf.reshape(tst_X, (tst_X.shape[0], tst_X.shape[1], 1))
        pred_Y_proba = model.predict(tst_X)


    end_time = time.time()
    
    if multiclass:
        auc_score, precision, recall, f1, accuracy, conf_matrix = calculateScoreMulticlass(tst_Y, pred_Y_proba)
    else:
        auc_score, precision, recall, f1, accuracy, conf_matrix = calculateScoreBinary(tst_Y, pred_Y_proba)
    
    if start_time and end_time:
        execution_time = end_time - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Total execution time: {int(hours)}h:{int(minutes)}m:{int(seconds)}s.")

    print(f"AUC Score: {auc_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"{auc_score:.4f},{precision:.4f},{recall:.4f},{f1:.4f},{accuracy:.4f}")
    print("Confusion Matrix with Labels:")
    print(conf_matrix)
    
    if args.heatmap:
        plt.figure(figsize=(7, 7))
        heatmap = sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt="d", 
            xticklabels=labels, 
            yticklabels=labels, 
            cmap="Oranges",
            cbar=False,
            square=True,
            annot_kws={"size": 18,} #"weight": "bold"}
        )
        heatmap.set_title(f"{chooseName(args.model_name)}", fontsize=26, loc='center', y=-0.15, fontweight="bold")
        heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)

        plt.subplots_adjust(bottom=0.15, top=0.95, left=0.05, right=0.95)
        plt.savefig(HEATMAPS_PATH + f"{args.model_name}_{args.train_path}_{args.test_path}_heatmap.png")

def testGridSearch(df, model, le):
    print("Mode GridSearchCV...")
    columns = list(df.columns)
    columns.remove("Label")
    X = df[columns]
    Y = df["Label"]
    
    Y = pd.Series(le.transform(Y))

    config_file = f"{CONFIG_PATH}/gridSearch_config.json"
    try:
        with open(config_file, "r") as file:
            grid_params = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"GridSearch configuration file for model '{args.model_name}' not found at {config_file}.")

    # We don't need logs from model itself
    if args.model_name not in ["dt", "ab"]: # Decision tree and AdaBoost have no verbose parameter
        model.set_params(verbose=0)

    grid_search = GridSearchCV(estimator=model, param_grid=grid_params[args.model_name], cv=3, scoring='f1_macro', n_jobs=12, verbose=1)
    grid_search.fit(X, Y)

    print(f"Best Parameters from Grid Search: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    return best_model

def kfoldCV(first_df, second_df, model, oe, k=5):
    print("Mode k-fold Cross-validation")
    df = pd.concat([first_df, second_df], axis=0, ignore_index=False)

    columns = list(df.columns)
    columns.remove("Label")
    df_features = df[columns]
    df_target = df["Label"].to_numpy().reshape(-1, 1)
    df_target = oe.transform(df_target).flatten()
    df_target = pd.Series(df_target, name="Label")
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_stats = {}
    avg_conf_matrix = np.zeros((len(df_target.unique()), len(df_target.unique())))  # Initialize empty matrix for averaging

    # model needs to be builded before loop as keras throws problems
    if args.model_name in ["mlp", "1dcnn"]:
        # this is true for merged awid2 and awid3, same problem that keras throws errors for passing tensor.shapes
        model = model([0, 33, 1], [0, 3])

    start_time = time.time()

    for fold, (train_index, test_index) in enumerate(kf.split(df_features), 1):
        trn_X, tst_X = df_features.iloc[train_index], df_features.iloc[test_index]
        trn_Y, tst_Y = df_target.iloc[train_index], df_target.iloc[test_index]

        if args.model_name not in ["1dcnn", "mlp"]:
            # Shallow learning path
            model.fit(trn_X, trn_Y)
        else: 
            # MLP / 1DCNN path - we change to tensors
            trn_X = np.array(trn_X).astype('float32')
            trn_Y = np.array(trn_Y).astype('float32')
            trn_X = tf.reshape(trn_X, (trn_X.shape[0], trn_X.shape[1], 1))
            trn_Y = tf.keras.utils.to_categorical(trn_Y, len(set(trn_Y)))

            early_stopping = EarlyStopping(patience=2)
            model.fit(trn_X, trn_Y, batch_size=128, validation_split=0.2, epochs=50, callbacks=[early_stopping])

        if args.model_name not in ["1dcnn", "mlp"]:
            # Shallow learning path
            if hasattr(model, "predict_proba"):
                pred_Y_proba = model.predict_proba(tst_X)
            else:
                # Handle models without predict_proba
                decision_scores = model.decision_function(tst_X)
                if multiclass:
                    pred_Y_proba = softmax(decision_scores, axis=1)
                else:
                    pred_Y_proba = softmax(np.column_stack((-decision_scores, decision_scores)), axis=1)
        else:
            # MLP / 1DCNN path - we change to tensors
            tst_X = np.array(tst_X).astype('float32')
            tst_Y = np.array(tst_Y).astype('float32')    
            tst_X = tf.reshape(tst_X, (tst_X.shape[0], tst_X.shape[1], 1))
            pred_Y_proba = model.predict(tst_X)

        # Calculate statistics
        if multiclass:
            auc, precision, recall, f1, accuracy, conf_matrix = calculateScoreMulticlass(tst_Y, pred_Y_proba)
        else:
            auc, precision, recall, f1, accuracy, conf_matrix = calculateScoreBinary(tst_Y, pred_Y_proba)

        fold_stats[fold] = {
            "AUC Score": auc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Accuracy": accuracy,
            "Confusion Matrix": conf_matrix
        }
        avg_conf_matrix += conf_matrix

        # Print fold results
        print(f"Fold {fold} Results:")
        print(f"AUC Score: {auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("-" * 40)

    end_time = time.time()

    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total execution time: {int(hours)}h:{int(minutes)}m:{int(seconds)}s.")

    avg_stats = {metric: np.mean([stats[metric] for stats in fold_stats.values()]) for metric in fold_stats[1].keys()}
    print("Average Statistics Across Folds:")
    for metric, avg_value in avg_stats.items():
        print(f"Mean {metric}: {avg_value:.4f}")
    print(",".join([f"{avg_value:.4f}" for metric, avg_value in avg_stats.items()]))
    
    avg_conf_matrix /= k
    np.set_printoptions(precision=1, suppress=True)
    avg_conf_matrix = np.round(avg_conf_matrix, 1)

    print("Average Confusion Matrix:")
    print(avg_conf_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test models, use -gs for GridSearchCV, use -hm to generate heatmap")
    
    parser.add_argument("train_path", type=str, help="Path to the training dataset (e.g. AWID2 or AWID3).")
    parser.add_argument("test_path", type=str, help="Path to the testing dataset (e.g. AWID2 or AWID3).")
    parser.add_argument("model_name", type=str, help="Name of the model to train (e.g. lgbm).")
    parser.add_argument("-gs", "--gridsearch", required=False, action="store_true", help="Perform GridSearchCV for hyperparameter tuning.")
    parser.add_argument("-hm", "--heatmap", required=False, action="store_true", help="Saves heatmap to .png file")
    parser.add_argument("-cv", "--crossval", required=False, action="store_true", help="Changes evaluation method to k-fold Cross Validation")
    args = parser.parse_args()

    train_path = decideOnPath(args.train_path)
    test_path = decideOnPath(args.test_path)
    train_df = loadDataset(train_path)
    test_df = loadDataset(test_path)

    # print(train_df["Label"].value_counts())
    # print(test_df["Label"].value_counts())

    train_classes_num = len(train_df["Label"].unique())
    test_classes_num = len(test_df["Label"].unique())

    if train_classes_num != test_classes_num:
        raise ValueError(f"Datasets num of classes should match, train classes: {train_classes_num}, test classes {test_classes_num}")

    global multiclass; multiclass = True
    if train_classes_num == 2:
        multiclass = False

    le = OrdinalEncoder()

    if args.test_path in ["AWID2", "AWID3"]:
        le = OrdinalEncoder(categories=[["Normal", "Impersonation", "Flooding"]])
    elif args.train_path in ["AWID2_BIN", "AWID3_BIN"]:
        le = OrdinalEncoder(categories=[["Normal", "Attack"]])
    elif args.test_path in ["AWID2_NF", "AWID3_NF"]:
        le = OrdinalEncoder(categories=[["Normal", "Flooding"]])
    le.fit(np.array(le.categories).reshape(-1, 1))

    model = getModel(args.model_name)

    if args.crossval:
        kfoldCV(train_df, test_df, model, le)
    else:
        model = train(train_df, model, le)
        test(test_df, model, le)
