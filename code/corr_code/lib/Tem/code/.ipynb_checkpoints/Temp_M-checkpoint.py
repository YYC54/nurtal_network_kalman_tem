import sys

sys.path.append(".")
from utils.logging_info import Logger
import logging
from datetime import datetime
import pandas as pd
from lib.Tem.code.utils import func
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from pykalman import KalmanFilter
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import mean_absolute_error
import warnings  # Supress warnings
from configs import config

warnings.filterwarnings("ignore")
Logger = Logger(path="./log/tem-corr.log", clevel=logging.ERROR, Flevel=logging.DEBUG)
import os



class Temp_M(object):
    """xxx 订正方法类
    self.factime    :预测时间
    self.dataframe  :输入变量
    self.eleent     :
    self.methods    :
    self.train_data :训练数据

    """

    def __init__(self, factime, dataframe, element, methods):
        self.factime = factime
        self.dataframe = dataframe
        self.eleent = element
        self.methods = methods
        self.train_data = None
        self.current_time = datetime.now()
        self.feats_name = [
            "hour",
            "700q",
            "50u",
            "100r",
            "1000w",
            "70u",
            "850q",
            "tcwv",
            "400q",
            "v10_min_8",
            "300t",
            "850v",
            "800r",
            "deg0l",
            "1000t_s1",
            "tpe",
            "700t",
            "20t",
            "400gh",
            "mx2t3",
            "925u",
            "mn2t3",
            "cape",
            "50gh",
            "20r",
            "500t",
            "900gh",
            "20gh",
            "1000t_max_8",
            "v10_s1",
            "v10",
            "u10",
            "10u",
            "600u",
            "fal",
            "925r",
            "500u",
            "250u",
            "900t",
            "950q",
            "925v",
            "850gh",
            "cp",
            "900w",
            "250r",
            "100gh",
            "v100",
            "950t",
            "1000r",
            "1000q",
            "850t",
            "950u",
            "925q",
            "950r",
            "400r",
            "200q",
            "600t",
            "t2m_avg_8",
            "500gh",
            "900r",
            "200u",
            "950gh",
            "t2m",
            "1000gh",
            "800u",
            "850u",
            "fg310",
            "t2m_min_8",
            "tcc",
            "200v",
            "100v",
            "800gh",
            "20u",
            "150pv",
            "250t",
            "d2m",
            "v10_avg_8",
            "10gh",
            "300u",
            "800w",
            "1000t_var_8",
            "900v",
            "70gh",
            "400t",
            "100u",
            "950v",
            "400u",
            "600r",
            "msl",
            "925gh",
            "rsn",
            "150u",
            "600v",
            "900q",
            "300r",
            "600q",
            "250q",
            "850w",
            "200r",
            "1000t",
            "250pv",
            "10t",
            "t2m_s1",
            "500q",
            "skt",
            "150v",
            "100t",
            "sd",
            "500r",
            "tcw",
            "800v",
            "150r",
            "950w",
            "t2m_max_8",
            "925t",
            "600gh",
            "capes",
            "250gh",
            "300q",
            "800t",
            "150q",
            "1000t_avg_8",
            "1000u",
            "850r",
            "sf",
            "200gh",
            "100q",
            "900u",
            "300gh",
            "200t",
            "800q",
            "700u",
            "600pv",
            "200pv",
            "925w",
            "1000v",
            "150gh",
            "50q",
            "250v",
            "10r",
            "1000t_min_8",
            "level",
            "time",
            "dtime",
            "stime",
            "id",
            "lon",
            "lat",
            "time_order",
        ]
        self.scaler_path = config["tem"]["scaler_path"]
        self.model_tem = config["tem"]["save_model_path"]
        self.train_data_path = config["tem"]["train_data_path"]
        # self.final_csv = './lib/Tem/save/predictions.csv'

    def _feature(self):
        """特征工程
        :return:
        self.train_data:训练数据集
        self.dataframe:预测数据集
        """

        Logger.info(f"特征工程-开始特征工程")
        # try:
        data = pd.read_csv(self.train_data_path)
        self.train_data = func.data_process(data)
        subset = [
            "level",
            "time",
            "dtime",
            "id",
            "lon",
            "lat",
            "stime",
            "v100",
            "tcc",
            "sf",
            "msl",
            "lsp",
            "cape",
            "sst",
            "lcc",
            "sd",
            "cp",
            "u100",
            "skt",
            "tpe",
            "sp",
            "d2m",
            "u10",
            "v10",
            "t2m",
            "p3020",
            "tcwv",
            "rsn",
            "deg0l",
            "tcw",
            "fal",
            "fzra",
            "capes",
            "1000t",
            "950t",
            "925t",
            "900t",
            "850t",
            "800t",
            "700t",
            "600t",
            "500t",
            "400t",
            "300t",
            "250t",
            "200t",
            "150t",
            "100t",
            "70t",
            "50t",
            "20t",
            "10t",
            "mx2t3",
            "fg310",
            "mn2t3",
            "1000gh",
            "950gh",
            "925gh",
            "900gh",
            "850gh",
            "800gh",
            "700gh",
            "600gh",
            "500gh",
            "400gh",
            "300gh",
            "250gh",
            "200gh",
            "150gh",
            "100gh",
            "70gh",
            "50gh",
            "20gh",
            "10gh",
            "1000u",
            "950u",
            "925u",
            "900u",
            "850u",
            "800u",
            "700u",
            "600u",
            "500u",
            "400u",
            "300u",
            "250u",
            "200u",
            "150u",
            "100u",
            "70u",
            "50u",
            "20u",
            "10u",
            "1000v",
            "950v",
            "925v",
            "900v",
            "850v",
            "800v",
            "700v",
            "600v",
            "500v",
            "400v",
            "300v",
            "250v",
            "200v",
            "150v",
            "100v",
            "70v",
            "50v",
            "20v",
            "10v",
            "1000r",
            "950r",
            "925r",
            "900r",
            "850r",
            "800r",
            "700r",
            "600r",
            "500r",
            "400r",
            "300r",
            "250r",
            "200r",
            "150r",
            "100r",
            "70r",
            "50r",
            "20r",
            "10r",
            "1000w",
            "950w",
            "925w",
            "900w",
            "850w",
            "800w",
            "700w",
            "600w",
            "500w",
            "400w",
            "300w",
            "250w",
            "200w",
            "150w",
            "100w",
            "70w",
            "50w",
            "20w",
            "10w",
            "1000q",
            "950q",
            "925q",
            "900q",
            "850q",
            "800q",
            "700q",
            "600q",
            "500q",
            "400q",
            "300q",
            "250q",
            "200q",
            "150q",
            "100q",
            "70q",
            "50q",
            "20q",
            "10q",
            "1000d",
            "950d",
            "925d",
            "900d",
            "850d",
            "800d",
            "700d",
            "600d",
            "500d",
            "400d",
            "300d",
            "250d",
            "200d",
            "150d",
            "100d",
            "70d",
            "50d",
            "20d",
            "10d",
            "1000pv",
            "950pv",
            "925pv",
            "900pv",
            "850pv",
            "800pv",
            "700pv",
            "600pv",
            "500pv",
            "400pv",
            "300pv",
            "250pv",
            "200pv",
            "150pv",
            "100pv",
            "70pv",
            "50pv",
            "20pv",
            "10pv",
            "kyi",
            "ptype",
            "KI",
            "SI",
            "TT",
            "LI",
            "SWEAT",
            "BLI",
            "storm",
            "time_order",
            "ct",
            "wd",
            "ws",
            "wp",
            "tp",
            "dp",
            "ps",
            "rh",
            "pp",
            "td",
            "sw_max",
            "sw_avg",
            "vis",
            "sw_max1",
            "sw_avg1",
            "sw_max2",
            "sw_avg2",
        ]
        self.train_data = self.train_data.dropna(subset=subset)
        self.train_data["time"] = pd.to_datetime(self.train_data["time"])
        self.train_data = func.process_outlier(self.train_data, subset)
        self.train_data = func.get_time_features(self.train_data)
        ori_feats = ["t2m", "v10", "u10", "1000t"]
        self.train_data = func.get_window(self.train_data, ori_feats)
        self.train_data = func.get_lag_features(self.train_data, ori_feats)

        # except Exception as e:
        #     Logger.error(f'特征工程-未生成训练数据集- {self.factime}')
        # try:
        # self.dataframe = self.dataframe.dropna(subset=subset)
        try:
            self.dataframe["stime"] = pd.to_datetime(
                self.dataframe["stime"], format="%Y/%m/%d %H:%M:%S"
            )
        except Exception as e:
            self.dataframe["stime"] = pd.to_datetime(
                self.dataframe["stime"], format="ISO8601"
            )
        self.dataframe["time"] = pd.to_datetime(self.dataframe["time"])
        self.dataframe = func.get_time_features(self.dataframe)
        self.dataframe = func.get_window(self.dataframe, ori_feats)
        self.dataframe = func.get_lag_features(self.dataframe, ori_feats)
        self.dataframe["time_order"] = self.dataframe["stime"].dt.hour
        # except Exception as e:
        #     Logger.error(f'特征工程-未生成预测数据集- {self.factime}')

        return self.train_data, self.dataframe

    def _train(self):
        """训练
        :return:
        self.model_tem:模型
        """
        Logger.info(f"神经网络模型训练-神经网络方法-开始训练")
        try:
            features = self.train_data[
                [feat for feat in self.feats_name if feat != "tp"]
            ]
            target = self.train_data["tp"]

            features["time"] = pd.to_datetime(features["time"])
            features["time"] = features["time"].apply(lambda x: x.timestamp())
            features["stime"] = pd.to_datetime(features["time"])
            features["stime"] = features["stime"].apply(lambda x: x.timestamp())

            features, target = shuffle(features, target, random_state=0)
            # 80%作为训练集，20%作为测试集
            x_train, x_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            joblib.dump(scaler, self.scaler_path)

            num_cols = x_train.shape[1]
            model = Sequential(
                [
                    Dense(512, activation="tanh", input_shape=(num_cols,)),
                    # Dropout(0.2), # 20% 的神经元在每次训练迭代中被置为 0
                    Dense(256, activation="sigmoid"),
                    # Dropout(0.2), # 20% 的神经元在每次训练迭代中被置为 0
                    Dense(128, activation="tanh"),
                    Dense(64, activation="sigmoid"),
                    Dense(32, activation="tanh"),
                    Dense(16, activation="tanh"),
                    Dense(8, activation="tanh"),
                    Dense(1, activation="linear"),
                ]
            )

            # Adam优化器
            optimizer = Adam(lr=0.001)
            # 随机梯度下降

            model.compile(optimizer=optimizer, loss=func.quantile_loss, metrics=["mse"])

            early_stopping = EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )

            lr_scheduler = LearningRateScheduler(func.scheduler)

            callbacks = [early_stopping, lr_scheduler]

            history = model.fit(
                x_train, y_train, epochs=2000, validation_split=0.2, callbacks=callbacks
            )

            test_loss, test_mae = model.evaluate(x_test, y_test)

            print("Test Mean Absolute Error (MAE): {:.2f}".format(test_mae))

            model.save(self.model_tem)
        except Exception as e:
            Logger.error(f"训练-Corr方法-训练失败- {self.current_time}" f"错误信息为：{e}")
        return self.model_tem

    def _pred(self):
        """预测

        :return:
        """
        Logger.info(f"开始预测")

        # try:
        model = func.load_pretrained_model(self.model_tem)
        self.dataframe = self.dataframe.query("id==2")
        new_features_scaled = func.preprocess_data(
            self.dataframe, self.scaler_path, self.feats_name
        )
        predictions = func.predict_values(model, new_features_scaled)
        predictions_df = pd.DataFrame()

        for col in ["time", "time_order", "dtime", "id", "lon", "lat", "stime", "t2m"]:
            predictions_df[col] = self.dataframe[col]
        predictions_df["t2m"] = predictions_df["t2m"] - 273.15
        predictions_df["北京时间"] = predictions_df["time"] + pd.Timedelta(hours=8)
        predictions_df["tem_neural_network"] = predictions

        # predictions_df.to_csv(
        #     self.final_csv,
        #     index=False)
        # except Exception as e:
        #     Logger.error(f'预测-神经网络方法-神经网络方法预测失败- {self.factime}')
        filtered_predictions_df = pd.DataFrame()
        # try:
        # 对每个起始时间应用卡尔曼滤波器
        print("predictions_df.columns.values", predictions_df.columns.values)
        for stime, group in predictions_df.groupby("stime"):
            # 按照dtime排序，保证预报按照时间顺序
            group = group.sort_values("dtime")

            # predicted_temp = group['tem_neural_network'].values  # 获取当前时间序列的神经网络预测的温度值
            predicted_t2m = group["t2m"].values

            # initial_state = predicted_temp[0]  # 初始状态设为神经网络预测的第一个温度值
            initial_state = predicted_t2m[0]
            filtered_state_means = np.zeros(predicted_t2m.shape)  # 初始化一个用于存放滤波状态均值的数组
            filtered_state_covariances = np.zeros(
                predicted_t2m.shape
            )  # 初始化一个用于存放滤波状态协方差的数组
            global_error_variance = np.var(
                predicted_t2m - filtered_state_means
            )  # 计算一个全局误差方差

            kf = KalmanFilter(
                transition_matrices=[1],
                observation_matrices=[1],
                initial_state_mean=initial_state,
                initial_state_covariance=1,
                transition_covariance=np.var(predicted_t2m),
            )

            for t in range(len(predicted_t2m)):
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
                        observation=predicted_t2m[t],
                        observation_matrix=np.atleast_2d(1),
                        observation_covariance=0.5 * global_error_variance,
                    )

            # 将处理后的预报添加到新的DataFrame
            group["tem_kalman_filter"] = filtered_state_means
            filtered_predictions_df = pd.concat([filtered_predictions_df, group])

        # except Exception as e:
        #     Logger.error(f'预测-卡尔曼滤波方法-卡尔曼滤波方法预测失败- {self.factime}')

        filtered_predictions_df["ensemble"] = (
            filtered_predictions_df["tem_neural_network"]
            + filtered_predictions_df["tem_kalman_filter"]
        ) / 2

        # 保存包含滤波后预测的新DataFrame
        # filtered_predictions_df.to_csv(
        #             self.final_csv,
        #             index=False)

        return filtered_predictions_df


# if __name__ == "__main__":


#     # Logger.info('处理XXX-Corr方法-日志信息- 2020-01-02 00')
#     # '''
#     #     日志信息包括：（处理订正的时间，出错点）
#     # '''
#     dataframe = pd.read_csv('./lib/Tem/dataset/test.csv')

#     temp_m = Temp_M(factime= '2022-01-01', dataframe=dataframe, element='corr-Tem', methods = ("kalman_filter", "neural_network" , "ensemble"))
#     feature = temp_m._feature()
#     # train = temp_m._train()
#     corr_datarfarame = temp_m._pred()
