from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from pykalman import KalmanFilter
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# from keras.utils import multi_gpu_model
# from tensorflow.keras.utils import multi_gpu_model
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle


def data_process(df):
    try:
        df["stime"] = pd.to_datetime(df["stime"], format="%Y/%m/%d %H:%M:%S")
    except Exception as e:
        df["stime"] = pd.to_datetime(df["stime"], format="ISO8601")

    df["year"] = df["stime"].dt.year
    df["month"] = df["stime"].dt.month

    train_data = df[(df["lon"] == 102.03) & (df["lat"] == 28.23)]

    return train_data


def scheduler(epoch, lr):
    start_decay_epoch = 100
    end_decay_epoch = 200
    decay_rate = 0.1

    if start_decay_epoch <= epoch < end_decay_epoch:
        return lr * decay_rate
    else:
        return lr


def process_outlier(df, feats):
    _ = df[feats[8:]]
    # 1. 计算每个气象要素的平均值和标准差
    mean_values = _.mean()
    std_values = _.std()
    # 2. 根据三倍标准差法，设定异常值的阈值
    threshold = 3 * std_values
    # 3. 遍历每个气象要素的数值，将超过设定阈值的值标记为异常值
    is_outlier = (_ > mean_values + threshold) | (_ < mean_values - threshold)
    # 4. 对于标记为异常值的数据，可以根据需要选择删除、替换或进行插补处理
    # 假设你选择删除异常值
    df = df[~is_outlier.any(axis=1)]
    #     df = df.query('sw_max1<20 and sw_max2<20')
    return df


# 时间特征 *
def get_time_features(df):
    df["time"] = pd.to_datetime(df["time"])
    #     df['year'] = df.time.dt.year
    df["month"] = df.time.dt.month
    df["day"] = df.time.dt.day
    df["hour"] = df.time.dt.hour
    return df


# 构造窗口统计特征 *
def get_window(df, ori_feats):
    df = df.sort_values(["stime", "dtime"]).reset_index(drop=True)

    for feat in ori_feats:
        # window=8的原因是8为一个周期
        df[feat + "_max_8"] = (
            df.groupby("stime")[feat].rolling(window=8).max().reset_index(drop=True)
        )
        df[feat + "_avg_8"] = (
            df.groupby("stime")[feat].rolling(window=8).mean().reset_index(drop=True)
        )
        df[feat + "_var_8"] = (
            df.groupby("stime")[feat].rolling(window=8).var().reset_index(drop=True)
        )
        df[feat + "_min_8"] = (
            df.groupby("stime")[feat].rolling(window=8).min().reset_index(drop=True)
        )

    df = df.fillna(method="bfill")
    df = df.fillna(method="ffill")
    print(df)
    return df


# 构造时滞特征
def get_lag_features(df, ori_feats):
    df = df.sort_values(["stime", "dtime"]).reset_index(drop=True)

    for feat in ori_feats:
        df[feat + "_s1"] = df.groupby("stime")[feat].shift(1)

    # 构造后可能首尾出现空值
    df = df.fillna(method="bfill")
    df = df.fillna(method="ffill")

    return df


def select_features(df, thus):
    df_corr = df.drop_duplicates(subset=["time"])
    df_corr_1_res = pd.DataFrame(df_corr.corr()["tp"])
    df_corr_1_res.sort_values("tp", ascending=False)
    # df_corr_1_res.to_csv('tp_corr.csv')

    df_corr_1_res = df_corr_1_res.query("abs(tp)>=@thus")

    feats = list(set(df_corr_1_res.index))
    feats = list(
        filter(lambda x: x not in ["tp", "time", "stime", "time_order"], feats)
    )
    # print(f'corr>{thus}:{feats}')

    feats_name = [
        "level",
        "time",
        "dtime",
        "stime",
        "id",
        "lon",
        "lat",
        "time_order",
        "tp",
    ] + feats
    return feats_name


def quantile_loss(y_true, y_pred):
    q = 0.8  # 可以根据你的需要修改
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))


"""
预测函数
"""


def load_pretrained_model(model_path):
    return load_model(model_path, custom_objects={"quantile_loss": quantile_loss})


def preprocess_data(data, scaler_path, feats_name):
    features = data[[feat for feat in feats_name if feat != "tp"]]
    features["time"] = pd.to_datetime(features["time"])
    features["time"] = features["time"].apply(lambda x: x.timestamp())
    features["stime"] = pd.to_datetime(features["time"])
    features["stime"] = features["stime"].apply(lambda x: x.timestamp())

    scaler = joblib.load(scaler_path)
    features_scaled = scaler.transform(features)

    return features_scaled


def predict_values(model, features):
    return model.predict(features)
