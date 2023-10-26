# -*- coding:utf-8 -*-
import numpy as np
import metpy
from metpy import calc
from metpy.units import units
import pandas as pd
import pickle
from configs import config


class Feature_Engineering(object):
    def __init__(self, df, type):
        self.df = df
        self.type = type
        self.CONSTANT_FEATS = config['maxWins']['CONSTANT_FEATS']

        # 挑选部分源特征做时滞、窗口统计特征
        self.ori_feats = [
            "wind_speed10",
            "wind_speed30",
            "wind_speed50",
            "wind_speed70",
            "wind_speed100",
            "wind_speed_L900",
            "wind_speed_L925",
            "wind_speed_L950",
            "wind_speed_L1000",
        ]
        self.int_features = [
            "sw_max1",
            "sw_max2",
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
            "fg310",
            "tcwv",
            "rsn",
            "deg0l",
            "mn2t3",
            "mx2t3",
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
        ]

    def get_data(self, df):
        df.time = pd.to_datetime(df["time"])
        try:
            df.stime = pd.to_datetime(df["stime"], format="ISO8601")
        except Exception as e:
            df.stime = pd.to_datetime(df["stime"])

        df["time_order"] = df.stime.dt.hour
        df["北京时间"] = df["time"] + pd.Timedelta(hours=8)
        df = df.query(f"dtime>=0 and dtime<=120").reset_index(drop=True)
        df = df.sort_values(["stime", "dtime"]).reset_index(drop=True)

        int_feats_name = self.CONSTANT_FEATS + self.int_features

        if self.type == "test":
            df = df.query("id==2")  # 只预测二站
            # df = df.rename(columns={'tp':'tpe','t':'10t'})
            int_feats_name = [
                x for x in int_feats_name if x not in ["sw_max1", "sw_max2"]
            ]

        df = df[int_feats_name]
        # 日志添加有效数据，空值数据多少条
        return df

    def process_outlier(self, df):
        _ = df.iloc[:, 8:]
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
        df = df.query("sw_max1<20 and sw_max2<20")
        return df

    # train/test
    # 合成风向风速
    def speed_direction_features(self, df):
        # 层级风分量
        levels = [10, 100, 1000, 950, 925, 900, 500, 700]
        for level in levels:
            u_col = f"{level}u"
            v_col = f"{level}v"

            df[f"wind_speed_L{level}"] = metpy.calc.wind_speed(
                df[u_col].values * units("m/s"), df[v_col].values * units("m/s")
            )
            df[f"wind_direction_L{level}"] = metpy.calc.wind_direction(
                df[u_col].values * units("m/s"), df[v_col].values * units("m/s")
            )

        # 非层级风分量
        df["wind_speed10"] = metpy.calc.wind_speed(
            df["u10"].values * units("m/s"), df["v10"].values * units("m/s")
        )
        df["wind_direction10"] = metpy.calc.wind_direction(
            df["u10"].values * units("m/s"), df["v10"].values * units("m/s")
        )
        df["wind_speed100"] = metpy.calc.wind_speed(
            df["u100"].values * units("m/s"), df["v100"].values * units("m/s")
        )
        df["wind_direction100"] = metpy.calc.wind_direction(
            df["u100"].values * units("m/s"), df["v100"].values * units("m/s")
        )

        return df

    # train/test
    # 根据边界层理论计算其他风层
    def get_other_height_speed(self, df):
        levels = [30, 50, 70]
        for level in levels:
            df[f"wind_speed{level}"] = df["wind_speed10"] * (level / 10) ** (1 / 7)
        return df

    # 相邻风层计算风速差
    def get_height_speed_diff(self, df):
        df["wind_speed30_diff"] = df["wind_speed30"] - df["wind_speed10"]
        df["wind_speed50_diff"] = df["wind_speed50"] - df["wind_speed30"]
        df["wind_speed70_diff"] = df["wind_speed70"] - df["wind_speed50"]
        df["wind_speed100_diff"] = df["wind_speed100"] - df["wind_speed70"]

        df["wind_speed_L925_diff"] = df["wind_speed_L925"] - df["wind_speed_L900"]
        df["wind_speed_L950_diff"] = df["wind_speed_L950"] - df["wind_speed_L925"]
        df["wind_speed_L1000_diff"] = df["wind_speed_L1000"] - df["wind_speed_L950"]

        return df

    # 构造一阶时滞特征
    def get_lag_features(self, df):
        df = df.sort_values(["stime", "dtime"]).reset_index(drop=True)
        for feat in self.ori_feats:
            df[feat + "_s1"] = df.groupby("stime")[feat].shift(1)
        # 产生较多空值再删除加会不会损坏精度？
        return df

    # 构造窗口统计特征
    def get_window(self, df):
        df = df.sort_values(["stime", "dtime"]).reset_index(drop=True)

        for feat in self.ori_feats:
            df[feat + "_max_8"] = (
                df.groupby("stime")[feat].rolling(window=8).max().reset_index(drop=True)
            )
            df[feat + "_avg_8"] = (
                df.groupby("stime")[feat]
                .rolling(window=8)
                .mean()
                .reset_index(drop=True)
            )
            df[feat + "_var_8"] = (
                df.groupby("stime")[feat].rolling(window=8).var().reset_index(drop=True)
            )
        # 窗口特征可能会产生空值
        df = df.fillna(method="bfill")
        df = df.fillna(method="ffill")

        return df

    # 加入时间特征
    def get_time_features(sefl, df):
        # df['year'] = df.time.dt.year
        df["month"] = df.time.dt.month
        df["day"] = df.time.dt.day
        df["hour"] = df.time.dt.hour
        return df

    # 根据相关性筛选特征
    def select_features(self, df, thus):
        if self.type == "train":
            df_corr = df.drop_duplicates(subset=["time"])
            df_corr_1_res = pd.DataFrame(df_corr.corr()["sw_max1"])
            df_corr_1_res.sort_values("sw_max1", ascending=False)
            df_corr_1_res.to_csv(config["maxWins"]["corr_sw_max1_path"])

            df_corr_2_res = pd.DataFrame(df_corr.corr()["sw_max2"])
            df_corr_2_res.sort_values("sw_max2", ascending=False)
            df_corr_2_res.to_csv(config["maxWins"]["corr_sw_max2_path"])

            df_corr_1_res = df_corr_1_res.query("abs(sw_max1)>=@thus")
            df_corr_2_res = df_corr_2_res.query("abs(sw_max2)>=@thus")

            feats = list(
                set(list(df_corr_1_res.index)).union(set(list(df_corr_2_res.index)))
            )
            # 保持列的顺序
            feats = [
                x for x in feats if x not in self.CONSTANT_FEATS + ["sw_max1", "sw_max2"]
            ]

            selected_feats_name = self.CONSTANT_FEATS + ["sw_max1", "sw_max2"] + feats
            # 将筛选的特征名存储以便推理使用
            with open(config["maxWins"]["selected_feats_path"], "wb") as f:
                pickle.dump(self.CONSTANT_FEATS + feats, f)

        else:
            with open(config["maxWins"]["selected_feats_path"], "rb") as f:
                selected_feats_name = pickle.load(f)

        return df[selected_feats_name]

    def process_null(self, df):
        if self.type == "train":
            print("shape of raw train data:", df.shape)
            df = df.dropna()
            print("shape of deleted-null train data:", df.shape)
        else:
            df = df.fillna(method="pad")
            df = df.fillna(method="bfill")
            df = df.fillna(method="ffill")
        return df

    def process_data(self):
        df = self.get_data(self.df)
        if self.type == "train":
            df = self.process_outlier(df)  # 处理异常值
        df = self.speed_direction_features(df)  # 合成风向、风速
        df = self.get_other_height_speed(df)  # 计算其他高度层的风速
        df = self.get_height_speed_diff(df)  # 相邻高度风速差
        df = self.get_time_features(df)  # 加入时间特征
        df = self.get_window(df)  # 加入窗口统计特征
        df = self.get_lag_features(df)  # 时滞特征无效
        # 设定特征筛选阈值
        df = self.select_features(df, 0.1)  # 特征筛选
        df = self.process_null(df)
        self.df = df
        return self.df
