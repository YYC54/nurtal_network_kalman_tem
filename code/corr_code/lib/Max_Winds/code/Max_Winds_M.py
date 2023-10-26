#!/usr/bin/env python

# -*- coding:utf-8 -*-
"""
@author: zhangym
@contact: 976435584@qq.com
@software: PyCharm
@file: testM.py
@time: 2023/7/28 8:46
@Describe 
@Version 1.0
"""
import os
import sys
sys.path.append(".")

from utils.logging_info import Logger
from lib.Max_Winds.code.Feature_Engineering import Feature_Engineering
from lib.Max_Winds.code.Neural_Network import Neural_Network
import logging
import pandas as pd
import pickle
# from configs import config
import configparser
config = configparser.ConfigParser()
config.read(os.path.join(sys.path[0], 'config.ini'))

Logger = Logger(path="./log/swmax-corr.log", clevel=logging.ERROR, Flevel=logging.DEBUG)


class Max_Winds_M(object):
    """浅层风的订正方法类"""

    def __init__(self, factime, dataframe, element, methods):
        self.factime = factime
        self.dataframe = dataframe
        self.eleent = element
        self.methods = methods
        self.train_data_path = config.get('MaxWins_Path', 'train_data_path')

    def _feature(self, df, type):
        fe = Feature_Engineering(df, type)
        df = fe.process_data()

        return df

    def _train(self):
        train_df = pd.read_csv(self.train_data_path)
        train_df = self._feature(train_df, "train")
        sw_max_ids = [1, 2]
        nn = Neural_Network("train", train_df, sw_max_ids)
        nn.train_model()

    def _pred(self):
        # df = pd.read_csv('./lib/Max_Winds/dataset/qiancengfeng_leibao.csv')
        # df = df[[x for x in df.columns if x not in ['sw_max1', 'sw_max2']]][-1000:]
        df = self.dataframe
        df = self._feature(df, "test")
        # print(df)
        sw_max_ids = [1, 2]
        nn = Neural_Network("test", df, sw_max_ids)
        df_corrres = nn.pred_model()
        #         df_corrres.to_csv('./lib/Max_Winds/result/save/prediction.csv', index=False)
        return df_corrres


# if __name__ == "__main__":
#     Logger.info("处理XXX-Corr方法-日志信息- 2020-01-02 00")
#     """
#         日志信息包括：（处理订正的时间，出错点）
#     """
#     dataframe = pd.read_csv("./test.csv")
#     max_winds_M = Max_Winds_M(
#         factime="2020-01-02",
#         dataframe=dataframe,
#         element="result-MaxWins",
#         methods=("kalman_filter", "neural_network", "ensemble"),
#     )
#     max_winds_M._train('neural_network')
#     df_corrres = max_winds_M._pred()
