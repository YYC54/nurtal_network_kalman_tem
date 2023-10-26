# -*- coding:utf-8 -*-
import pickle
import sys

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Add, Input, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import numpy as np
from pykalman import KalmanFilter
import configparser
config = configparser.ConfigParser()
config.read(os.path.join(sys.path[0], 'config.ini'))


# 训练时gpu是否可用应该加入日志


class Neural_Network(object):
    def __init__(self, type, df, sw_max_ids):
        self.type = type
        self.df = df
        self.sw_max_ids = sw_max_ids
        self.CONSTANT_FEATS = [
                "time",
                "time_order",
                "dtime",
                "id",
                "lon",
                "lat",
                "stime",
                "北京时间",
            ]

    def data_loader(self, df):
        TRAIN_DATA = []
        VAL_DATA = []
        TEST_DATA = []
        data = dict()

        if self.type == "train":
            # 划分
            TRAIN_VAL_DATA = df.loc[:, "sw_max1":].sample(
                frac=1, random_state=0
            )  # shuffle

            TRAIN_DATA = TRAIN_VAL_DATA.iloc[: int(len(TRAIN_VAL_DATA) * 0.9)]
            VAL_DATA = TRAIN_VAL_DATA.iloc[int(len(TRAIN_VAL_DATA) * 0.9) :]

            # 归一化
            scaler = StandardScaler()
            TRAIN_DATA = scaler.fit_transform(TRAIN_DATA)
            VAL_DATA = scaler.transform(VAL_DATA)

            with open(config.get('MaxWins_Path', 'scaler_path'), "wb") as f:
                pickle.dump(scaler, f)

            data["train_X"] = TRAIN_DATA[:, 2:]
            data["train_y_1"] = TRAIN_DATA[:, 0]
            data["train_y_2"] = TRAIN_DATA[:, 1]

            data["val_X"] = VAL_DATA[:, 2:]
            data["val_y_1"] = VAL_DATA[:, 0]
            data["val_y_2"] = VAL_DATA[:, 1]
        else:
            TEST_DATA = df.copy()
            data["test_X"] = TEST_DATA[
                [x for x in df.columns if x not in self.CONSTANT_FEATS]
            ]
            # scaler包含“sw_max1,sw_max2”,因此跳过两位
            with open(config.get('MaxWins_Path', 'scaler_path'), "rb") as f:
                scaler = pickle.load(f)
            selected_mean = scaler.mean_[2:]
            selected_scale = scaler.scale_[2:]
            # 归一化测试数据
            data["test_X"] = (data["test_X"] - selected_mean) / selected_scale

        return TRAIN_DATA, VAL_DATA, TEST_DATA, data

    def quantile_loss(self, y_true, y_pred):
        error = y_true - y_pred
        q = 0.01
        return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))

    def build_network(self, sw_max_id, num_blocks=3, input_features=None):
        inputs = Input(shape=(input_features,), name="input")
        x = Dense(512, activation="relu", name="hidden_1")(inputs)

        for i in range(num_blocks):
            residual = x
            x = Dense(1024, activation="relu", name=f"hidden_{i + 2}_1")(x)
            x = Dense(512, activation="linear", name=f"hidden_{i + 2}_2")(x)
            x = Dropout(0.3, name=f"dropout_{i + 2}")(x)
            x = Add()([x, residual])
        prediction = Dense(1, activation="linear", name="final")(x)
        model = Model(inputs=inputs, outputs=prediction)
        opt = keras.optimizers.Adam(learning_rate=0.0001)

        # 定义百分位值
        if sw_max_id == 1:
            model.compile(optimizer=opt, loss="mae")
        else:
            q = 0.01
            model.compile(optimizer=opt, loss=self.quantile_loss)

        return model

    def lr_scheduler(self, epoch, lr):
        if epoch % 10 == 0 and epoch > 0:
            lr = lr * 0.5
        return lr

    def kalman_sm_max(self, target, df_res):
        kalman_res = pd.DataFrame()

        # 对每个起始时间应用卡尔曼滤波器
        for stime, group in df_res.groupby("stime"):
            # 按照dtime排序，保证预报按照时间顺序
            group = group.sort_values("dtime")
            predicted_sw_max = group[target + "_nn"].values  # 获取当前时间序列的神经网络预测的温度值
            initial_state = predicted_sw_max[0]  # 初始状态设为神经网络预测的第一个温度值
            filtered_state_means = np.zeros(
                predicted_sw_max.shape
            )  # 初始化一个用于存放滤波状态均值的数组
            filtered_state_covariances = np.zeros(
                predicted_sw_max.shape
            )  # 初始化一个用于存放滤波状态协方差的数组
            global_error_variance = np.var(
                predicted_sw_max - filtered_state_means
            )  # 计算一个全局误差方差

            kf = KalmanFilter(
                transition_matrices=[1],
                observation_matrices=[1],
                initial_state_mean=initial_state,
                initial_state_covariance=1,
                transition_covariance=np.var(predicted_sw_max),
            )

            for t in range(len(predicted_sw_max)):
                try:
                    if t == 0:
                        # 第一个点，用神经网络初始预测值
                        filtered_state_means[t] = initial_state
                        filtered_state_covariances[t] = 1
                    else:
                        # 更新卡尔曼滤波
                        (
                            filtered_state_means[t],
                            filtered_state_covariances[t],
                        ) = kf.filter_update(
                            filtered_state_mean=filtered_state_means[t - 1],
                            filtered_state_covariance=filtered_state_covariances[t - 1],
                            observation=predicted_sw_max[t],
                            observation_matrix=np.atleast_2d(1),
                            observation_covariance=0.5 * global_error_variance,
                        )
                except:
                    if t == 0:
                        # 第一个点，用神经网络初始预测值
                        filtered_state_means[t] = initial_state
                        filtered_state_covariances[t] = 1
            # 将处理后的预报添加到新的DataFrame
            group[target + "_kalman"] = filtered_state_means
            kalman_res = pd.concat([kalman_res, group])
        return kalman_res

    def train_model(self):
        TRAIN_DATA, VAL_DATA, TEST_DATA, data = self.data_loader(self.df)
        for sw_max_id in self.sw_max_ids:
            input_features = data["train_X"].shape[1]
            model = self.build_network(
                sw_max_id, input_features=input_features, num_blocks=3
            )
            print("Network Structure")
            print(model.summary())
            print("Training Data Shape: " + str(data["train_X"].shape))

            lr_callback = keras.callbacks.LearningRateScheduler(self.lr_scheduler)
            es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
            history = model.fit(
                x=data["train_X"],
                y=data["train_y_" + str(sw_max_id)],
                batch_size=32,
                epochs=500,
                verbose=1,
                callbacks=[es_callback, lr_callback],
                validation_data=(data["val_X"], data["val_y_" + str(sw_max_id)]),
            )
            plt.clf()
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"], label="val_loss")
            plt.title("Model Loss")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.savefig(
                os.path.join(
                    config.get('MaxWins_Path', 'loss_fig_path'),
                    f"loss_history_sw_max{sw_max_id}.png",
                )
            )
            model.save(
                os.path.join(
                    config.get('MaxWins_Path', 'loss_fig_path'), f"Max_Wind_{sw_max_id}.h5"
                )
            )

    def pred_model(self):
        TRAIN_DATA, VAL_DATA, TEST_DATA, data = self.data_loader(self.df)
        df_res = TEST_DATA[self.CONSTANT_FEATS]

        for sw_max_id in self.sw_max_ids:
            if sw_max_id == 1:
                model = keras.models.load_model(
                    os.path.join(
                        config.get('MaxWins_Path', 'save_model_path'), f"Max_Wind_{sw_max_id}.h5"
                    )
                )
            else:
                model = keras.models.load_model(
                    os.path.join(
                        config.get('MaxWins_Path', 'save_model_path'), f"Max_Wind_{sw_max_id}.h5"
                    ),
                    custom_objects={"quantile_loss": self.quantile_loss},
                )

            y_hat = model.predict(data["test_X"])
            with open(config.get('MaxWins_Path', 'scaler_path'), "rb") as f:
                scaler = pickle.load(f)
            # print(y_hat)
            y_hat = scaler.inverse_transform(
                np.concatenate(
                    (y_hat.reshape(-1, 1), y_hat.reshape(-1, 1), data["test_X"]), axis=1
                )
            )[:, sw_max_id - 1]

            df_res[f"sw_max{sw_max_id}_pred_nn"] = y_hat
            df_res = self.kalman_sm_max(f"sw_max{sw_max_id}_pred", df_res)
            df_res[f"sw_max{sw_max_id}_pred_ensemble"] = (
                df_res[f"sw_max{sw_max_id}_pred_nn"]
                + df_res[f"sw_max{sw_max_id}_pred_kalman"]
            ) / 2
            # print(df_res)
        return df_res
