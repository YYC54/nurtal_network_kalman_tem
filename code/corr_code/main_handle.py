#!/usr/bin/env python

# -*- coding:utf-8 -*-
"""
@author: zhangym
@contact: 976435584@qq.com
@software: PyCharm
@file: main.py
@time: 2023/6/9 10:52
@Describe 
@Version 1.0
"""
import sys
from collections import OrderedDict
import pandas as pd
import os
from lib.Max_Winds.code import Max_Winds_M
import lib.Max_Winds.code.tools as tools
from lib.Tem.code import Temp_M
from lib.Thunder.code import Thunder_M
from lib.Precipitation.code import Precipitation_M
import datetime
# from configs import config
import tensorflow as tf
import configparser

base_path = sys.path[0]
config = configparser.ConfigParser()
config.read(os.path.join(base_path, 'config.ini'))
# GPU设置
os.environ["CUDA_VISIBLE_DEVICES"] = config.get('CUDA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

CORR_BASE_PATH = config.get('CORR_BASE_PATH', 'CORR_BASE_PATH')  # 订正保存产品数据
STA_BASE_PATH = config.get('STA_BASE_PATH', 'STA_BASE_PATH')


class Corr_xichangsk(object):
    """订正"""

    def __init__(self, forcast_stime):
        self._variable_dict()
        self.forcast_stime = forcast_stime

    def __repr__(self):
        s = ["=========文件信息=========\n"]
        for k in self._head_level.keys():
            s.append("要素:{}: 订正方法 - {}\n".format(k, self._head_level[k]))
        return "".join(s)

    def _variable_dict(self):
        key = "result-Tem,result-MaxWins,result-Thunder,result-Pre".split(",")
        value = (
            '("kalman_filter","neural_network","ensemble");'
            '("kalman_filter","neural_network","ensemble");'
            '("random_forest","xgboost","neural_network","peiliao","ensemble");'
            '("frequency_match","neural_network","Precipitation_percentile","ensemble")'.split(";")
        )
        # key = "result-Pre".split(",")
        # value = (
        #     '("frequency_match","neural_network","Precipitation_percentile","ensemble")'.split(";")
        # )
        self._head_level = OrderedDict()
        for k, v in zip(key, value):
            self._head_level[k] = v

    def _gattar(self):
        """根据时间-获取数据
        :return dataframe
        """
        # 根据 self.forcast_stime ， 返回数据
        stime = datetime.datetime.strptime(self.forcast_stime, "%Y%m%d%H")
        Y = str(stime.year).zfill(4)
        Ym = str(stime.year).zfill(4) + str(stime.month).zfill(2)
        Ymd = (
                str(stime.year).zfill(4)
                + str(stime.month).zfill(2)
                + str(stime.day).zfill(2)
        )
        mdH = (
                str(stime.month).zfill(2)
                + str(stime.day).zfill(2)
                + str(stime.hour).zfill(2)
        )
        staData = pd.read_csv(os.path.join(STA_BASE_PATH, Y, Ym, Ymd, f"C1D{mdH}.csv"))
        staData = staData.rename(columns={"tp": "tpe", "t": "10t"})

        return staData

    def _corrA(self, staData, element, methods):

        if element == "result-Tem":
            temp_m = Temp_M.Temp_M(self.forcast_stime, staData, element, methods)
            temp_m._feature()
            if int(config.get('Tem_Path', 'is_train')):
                temp_m._train()
            corrresData = temp_m._pred()
            return corrresData

        if element == "result-Pre":
            methods = methods.replace('(', '').replace(')', '').replace('\"', '').split(',')
            precipition_m = Precipitation_M.Precipitation_M(
                factime=self.forcast_stime,
                element="result-6hPre",
                dataframe=None,
                methods=tuple(methods))
            precipition_m._feature()
            precipition_m._pred()

            corrresData = precipition_m.dataframe
            print(corrresData)
            corrresData.rename(columns={'frequency_match': 'fm', 'Precipitation_percentile': 'op',
                                        'neural_network': 'nm', 'ensemble': 'co'}, inplace=True)
            return corrresData
        elif element == "result-MaxWins":
            max_winds_m = Max_Winds_M.Max_Winds_M(
                self.forcast_stime, staData, element, methods
            )
            if int(config.get('MaxWins_Path', 'is_train')):
                max_winds_m._train()
            corrresData = max_winds_m._pred()
            return corrresData
        elif element == "result-Thunder":
            methods = methods.replace('(', '').replace(')', '').replace('\"', '').split(',')
            thunder_m = Thunder_M.Thunder_MODEL(
                self.forcast_stime, staData, element, tuple(methods)
            )

            if int(config.get('Thunder_Path', 'opt')) == 1:
                thunder_m._train()
            # else:
            corrresData = thunder_m._pred()

            return corrresData

    def _corr_data(self):
        """订正不同要素
        :return 订正结果
        """
        staData = self._gattar()
        if staData.empty:
            raise Exception("staDataFrame is None!")

        for k in self._head_level.keys():
            print("start result:", k, "方法：", self._head_level[k])  # 要素 ， 方法 ，
            corrresData = self._corrA(staData, element=k, methods=self._head_level[k])

            # 根据时间进行保存数据
            self._save(
                corresDateFrame=corrresData, element=k, methods=self._head_level[k]
            )
            print("finish ", k)

    def _save(self, corresDateFrame, element, methods):
        """保存订正数据"""

        time_forcast = pd.to_datetime(self.forcast_stime, format='%Y%m%d%H')
        Y_str = str(time_forcast.year).zfill(4)
        Ym_str = str(time_forcast.year).zfill(4) + str(time_forcast.month).zfill(2)
        Ymd_str = str(time_forcast.year).zfill(4) + str(time_forcast.month).zfill(2) + str(time_forcast.day).zfill(2)
        YmdH_str = str(time_forcast.year).zfill(4) + str(time_forcast.month).zfill(2) + str(time_forcast.day).zfill(
            2) + str(time_forcast.hour).zfill(2)
        if not corresDateFrame.empty:
            save_corr_path = os.path.join(
                CORR_BASE_PATH, Y_str, Ym_str, Ymd_str, YmdH_str, self.forcast_stime + "_" + element + ".csv"
            )
            if os.path.dirname(save_corr_path):
                if not os.path.exists(os.path.dirname(save_corr_path)):
                    os.makedirs(os.path.dirname(save_corr_path))
            if element == "result-MaxWins":
                # 浅层风特殊的保存方式
                tools.save_MaxWins(corresDateFrame, save_corr_path)
            else:
                corresDateFrame.to_csv(save_corr_path)
        else:
            print(
                "{} - {} element - {} methods is None , Please check the specific bug reason. ".format(
                    self.forcast_stime, element, methods
                )
            )


def main(dt):
    cxsk = Corr_xichangsk(forcast_stime=dt)
    print(cxsk)
    cxsk._corr_data()

# def main(i, d):
#     print(i, d)
#     print('hello_world!')


if __name__ == "__main__":
    # st = sys.argv[1]
    # et = sys.argv[2]
    st = '2023100800'
    et = '2023100912'
    st_time = datetime.datetime.strptime(st, '%Y%m%d%H')
    et_time = datetime.datetime.strptime(et, '%Y%m%d%H')

    date_list = pd.date_range(start=st_time, end=et_time, freq='12H').strftime('%Y%m%d%H')

    for i, date_one in enumerate(date_list):
        try:
            main(date_one)
        except Exception:
            print(i, date_one, '异常日期')
            continue
