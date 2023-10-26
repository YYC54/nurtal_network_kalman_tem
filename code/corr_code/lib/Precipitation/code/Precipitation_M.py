# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :046
# @File     :Precipitation_M.py
# @Date     :2023/8/17 9:45
# @Author   :HaiFeng Wang
# @Email    :bigc_HF@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import configparser
import copy
import datetime
import os
import pickle
import random
import numpy as np
import pandas as pd
import psycopg2
import sys

sys.path.append('/mnt/PRESKY/user/lihaifei/yuchen/46/corr_code/lib/Precipitation/code')
from lib.Precipitation.code.nn import RegressionModel
# import FM
from lib.Precipitation.code import FM
from lib.Precipitation.code import makePredata
from lib.Precipitation.code.xb.cmpt_adjust_to_optimal_percentile import adjust_to_opt_percentile

config = configparser.ConfigParser()

config.read(os.path.join(sys.path[0], 'config.ini'))


class Precipitation_M(object):
    def __init__(self, factime, dataframe, element, methods):
        self.factime = factime
        self.dataframe = dataframe
        self.eleent = element
        self.methods = methods
        self.data = None
        self.windows_size = 20

    def _feature(self):
        global mode_ec
        # real_time = datetime.datetime(2021, 7, 1, 0)+ datetime.timedelta(hours=-16
        real_time = pd.to_datetime(self.factime, format='%Y%m%d%H')

        t = real_time + datetime.timedelta(days=-1)

        start_time = t + datetime.timedelta(days=-self.windows_size)

        db_ip = config.get('Database', 'ip')
        db_post = config.get('Database', 'post')
        db_user = config.get('Database', 'user')
        db_password = config.get('Database', 'password')
        db_database_name = config.get('Database', 'database_name')
        db_table = config.get('Database', 'table')

        conn = psycopg2.connect(database=db_database_name, user=db_user,
                                password=db_password, host=db_ip, port=db_post)

        # conn = psycopg2.connect(database='app-046', user="postgres",
        #                         password="postgresBjhxkj", host="192.168.1.124", port="15434")

        df_sta = makePredata.conn_ht_data_center(conner=conn, table_name=db_table,
                                                 time=t + datetime.timedelta(hours=-16),
                                                 window=self.windows_size)
        print(f'>>>>>>>>>{df_sta}')
        '''
        # 合并预报数据逻辑
        mode_ec = pd.read_csv(
            r'/share/HPC/home/zhaox_hx/data/aidata/zhangym_tmp/DATA/ML_dataset/046_DataSet/qiancengfeng_leibao.csv')
        '''
        mode_ec = pd.DataFrame()
        d_l = pd.date_range(start_time, real_time, freq='12H')
        bp = config.get('STA_BASE_PATH', 'STA_BASE_PATH')
        for i, d in enumerate(d_l):
            try:
                y = str(d.year).zfill(4)
                ym = str(d.year).zfill(4) + str(d.month).zfill(2)
                ymd = str(d.year).zfill(4) + str(d.month).zfill(2) + str(d.day).zfill(2)
                mdH = str(d.month).zfill(2) + str(d.day).zfill(2) + str(d.hour).zfill(2)
                df = pd.read_csv(f'{os.path.join(bp, y, ym, ymd, f"C1D{mdH}.csv")}')
                if 't' in df.columns:
                    df.rename(columns={'t': '10t'}, inplace=True)
                cols = list(set(df.columns) - set(['kyi', '850kyi', '800kyi', '700kyi', '600kyi', '500kyi']))
                df = df[cols].query('id == 2')
                if not len(mode_ec):
                    mode_ec = df
                else:
                    mode_ec = pd.concat([mode_ec, df], ignore_index=True)

            except Exception:
                continue
        if mode_ec.empty:
            raise ('无数据')
        mode_ec['tpe6'] = np.NAN
        # mode_ec['stime'] = pd.to_datetime(mode_ec['stime'])
        try:
            mode_ec['stime'] = pd.to_datetime(mode_ec['stime'])
        except Exception as e:
            mode_ec["stime"] = pd.to_datetime(
                mode_ec["stime"], format="ISO8601"
            )
        # mode_ec['time'] = pd.to_datetime(mode_ec['time'])
        try:
            mode_ec['time'] = pd.to_datetime(mode_ec['time'])
        except Exception as e:
            mode_ec["time"] = pd.to_datetime(
                mode_ec["time"], format="ISO8601"
            )
        mode_ec = mode_ec.query(f'"{start_time}" <= stime <= "{real_time}"')
        ndata = makePredata.rt2p6(df_sta)
        data, pred = makePredata.pred2p6(ndata, mode_ec, real_time)
        if 'time_order' not in pred.columns:
            pred['stime'] = pd.to_datetime(pred['stime'])
            pred['time_order'] = pred['stime'].dt.hour
            data['stime'] = pd.to_datetime(data['stime'])
            data['time_order'] = data['stime'].dt.hour
        self.dataframe = pred
        self.data = data
        # mod by xgz {{{
        self.mode_ec = mode_ec
        # mod by xgz }}}

    def _train(self):
        pass

    def _pred(self):
        for func in self.methods:
            if func == 'frequency_match':
                dat = FM.main(tmp=self.data, data=self.dataframe, window_size=self.windows_size)
                self.dataframe[func] = dat[func]
                self.dataframe[func] = self.dataframe[func].fillna(self.dataframe['tpe6'])
            elif func == 'Precipitation_percentile':
                # mod by xgz {{{
                # op = copy.copy(self.dataframe)
                mode_ec = self.mode_ec
                mode_ec = mode_ec[mode_ec.time.isin(self.dataframe.time)]
                op = copy.copy(mode_ec)
                # mod by xgz }}}
                op.rename(columns={'tpe6': 'pred'}, inplace=True)
                abspath = config.get('Precipitation_Path', 'op_path')
                optimal_percentiles = pickle.load(open(abspath, 'rb'))
                df_adj = adjust_to_opt_percentile(op, optimal_percentiles)
                self.dataframe[func] = df_adj['pred_adj']
                self.dataframe[func] = self.dataframe[func].fillna(self.dataframe['tpe6'])
            elif func == 'neural_network':
                # 创建模型对象
                # 加载模型
                loaded_model = RegressionModel('pred')
                loaded_model.load_model(config.get('Precipitation_Path', 'nn_path'))
                predictions = loaded_model.predict(self.dataframe)
                self.dataframe[func] = predictions
                self.dataframe[func] = self.dataframe['tpe6']
            elif func == 'ensemble':
                self.dataframe[func] = self.dataframe.apply(lambda x: x['tpe6'] * 0.7 + x['frequency_match'] * 0.2 + x['Precipitation_percentile'] * 0.1, axis=1)
                self.dataframe[func] = self.dataframe[func].fillna(self.dataframe['tpe6'])
        self.dataframe['stime'] = pd.to_datetime(self.dataframe['stime'])
        self.dataframe['time'] = pd.to_datetime(self.dataframe['time'])
        self.dataframe['time_order'] = self.dataframe['stime'].dt.hour
        self.dataframe['北京时间'] = self.dataframe['time'].apply(lambda x: x + pd.Timedelta(8, unit='H'))
        self.dataframe = self.dataframe[
            ['time', 'time_order', 'dtime', 'id', 'lon', 'lat', 'stime', 'tpe6', 'frequency_match',
             'Precipitation_percentile', 'neural_network', 'ensemble', '北京时间']]


if __name__ == "__main__":
    p = Precipitation_M(
        factime="2023092100",
        element="result-6hPre",
        dataframe=None,
        # methods=["neural_network"])
        methods=("frequency_match", "Precipitation_percentile", "neural_network", "ensemble"))
    p._feature()
    p._pred()
    # print(p.dataframe[['tpe6', 'neural_network']])
    print(p.dataframe[['id', 'tpe6', 'frequency_match', 'Precipitation_percentile', 'ensemble']])

    # exit()
    # s = ['time', 'dtime', 'id', 'lon', 'lat', 'stime', 'pred', 'obs']
    # print(pred.columns)
    # exit()
    # print(pred.columns)
    # exit()
