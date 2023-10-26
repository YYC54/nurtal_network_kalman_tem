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
import configparser
import datetime
import os
import metpy.calc as mpcalc
from metpy.units import units
import math
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import meteva.method as mem
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import configparser
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from collections import Counter
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight


def SWEAT_calc(t500, t850, td850, win_spd500, win_spd850, win_dir500, win_dir850):
    """
    计算sweat指数SWEAT=12Td850+20(TT−49)+2f8+f5+125(S+0.2)
    """
    TT = t850 + td850 - 2 * t500
    f8 = win_spd850 * 2
    f5 = win_spd500 * 2
    S = math.sin(win_dir500 - win_dir850)
    if td850 < 0: td850 = 0
    if TT < 49: TT = 49
    if (130 < win_dir850 < 250) and (210 < win_dir500 < 310) and (win_dir500 - win_dir850 > 0) and (f8 >= 15) and (
            f5 >= 15):
        SS = 125 * (S + 0.2)
    else:
        SS = 0
    SWEAT = 12 * td850 + 20 * (TT - 49) + 2 * f8 + f5 + SS
    return SWEAT


def SWEAT_calc_loc(df, dpt, idx):
    u_500 = df.loc[idx, ['500u']].values[0]
    v_500 = df.loc[idx, ['500v']].values[0]
    u_850 = df.loc[idx, ['850u']].values[0]
    v_850 = df.loc[idx, ['850v']].values[0]
    win_spd500 = mpcalc.wind_speed(u_500 * units('m/s'), v_500 * units('m/s')).magnitude
    win_spd850 = mpcalc.wind_speed(u_850 * units('m/s'), v_850 * units('m/s')).magnitude
    win_dir500 = mpcalc.wind_direction(u_500 * units('m/s'), v_500 * units('m/s')).magnitude
    win_dir850 = mpcalc.wind_direction(u_850 * units('m/s'), v_850 * units('m/s')).magnitude
    t500 = df.loc[idx, ['500t']].values[0]
    t850 = df.loc[idx, ['850t']].values[0]
    td850 = dpt[4].magnitude

    return SWEAT_calc(t500, t850, td850, win_spd500, win_spd850, win_dir500, win_dir850)


class XXX_M(object):
    """ xxx 订正方法类 """

    def __init__(self, factime, dataframe, element, methods):
        self.factime = factime
        self.dataframe = dataframe
        self.eleent = element
        self.methods = methods

    @staticmethod
    def feature(a):
        """特征工程

        :return:
        """
        pass

    def train(self, X_train, y_train, X_valid, y_valid):
        """ 训练

        :return:
        """
        pass

    def pred(self):
        """ 预测

        :return:
        """
        pass


class Thunder(XXX_M):
    """ xxx 订正方法类 """

    def __init__(self, stime=None, order=None, element=None, methods=None, base_fea=None, target_fea=None, factime=None,
                 dataframe=None, params=None, element_model=None, model=None):
        super().__init__(factime, dataframe, element, methods)
        self.params = params
        self.element_model = element_model
        self.stime = stime
        self.order = order
        self.base_fea = base_fea
        self.target_fea = target_fea
        self.model = model

    def get_dataframe(self):
        """
        获取实时数据
        :return: 
        """
        bp = config.get('Realtime_Path', 'daily_path')
        stime_datetime = datetime.datetime.strptime(self.stime + self.order, '%Y%m%d%H')
        Y = str(stime_datetime.year).zfill(4)
        Ym = str(stime_datetime.year).zfill(4) + str(stime_datetime.month).zfill(2)
        Ymd = str(stime_datetime.year).zfill(4) + str(stime_datetime.month).zfill(2) + str(stime_datetime.day).zfill(2)
        dataframe = pd.read_csv(os.path.join(bp, Y, Ym, Ymd, f'C1D{self.stime[-4:]}{self.order}.csv'))
        self.dataframe = dataframe

    @staticmethod
    def feature(a):
        """特征工程
        for more: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#time-date-components
        """
        # self.get_dataframe()
        #
        # data = copy.copy(self.dataframe)

        a['time'] = pd.to_datetime(a['time'])


        try:
            a['stime'] = pd.to_datetime(a['stime'])
        except Exception as e:
            a['stime'] = pd.to_datetime(a['stime'],format="ISO8601")

        if 't' in a.columns:
            a.rename(columns={'t': '10t', 'tp': 'tpe'}, inplace=True)
        my_data = a.query('id == 2').reset_index(drop=True)
        
        # print(my_data.columns.values)

        if 'SWEAT' not in my_data.columns:
            p = [1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 20, 10]
            for idx in range(len(my_data)):
                ele = 't'
                ele_list = [str(x) + ele for x in p]

                t_list = (my_data.loc[idx, ele_list] - 273.15).values
                ele = 'q'
                ele_list = [str(x) + ele for x in p]
                r_list = (my_data.loc[idx, ele_list] * 1000).values

                # dpt = dewpoint_from_relative_humidity(list(t_list) * units.degC, list(r_list) * units.dimensionless)
                dpt = mpcalc.dewpoint_from_specific_humidity(p * units.hPa, list(t_list) * units.degC,
                                                             list(r_list) * units('g/kg'))

                # K指数
                my_data.loc[idx, ['KI']] = mpcalc.k_index(p * units.hPa, list(t_list) * units.degC, dpt).magnitude

                # SI
                my_data.loc[idx, ['SI']] = mpcalc.showalter_index(p * units.hPa, list(t_list) * units.degC,
                                                                  dpt).magnitude

                # TTI 总指数
                my_data.loc[idx, ['TT']] = mpcalc.total_totals_index(p * units.hPa, list(t_list) * units.degC,
                                                                     dpt).magnitude

                # prof
                prof = mpcalc.parcel_profile(p * units.hPa, t_list[0] * units.degC, dpt[0]).to('degC')

                # cape_cin
                # calculate surface based CAPE/CIN
                # cape, cin = mpcalc.cape_cin(p * units.hPa, list(t_list) * units.degC, dpt, prof)
                # my_data.loc[idx, ['cape_n']], my_data.loc[idx, ['cin']] = cape.magnitude, cin.magnitude

                # LI
                my_data.loc[idx, ['LI']] = mpcalc.lifted_index(p * units.hPa, list(t_list) * units.degC, prof).magnitude

                # sweat
                my_data.loc[idx, ['SWEAT']] = SWEAT_calc_loc(my_data, dpt, idx)

        # 提取月份信息
        # my_data = my_data[self.base_fea]
        my_data['month'] = pd.to_datetime(my_data['stime']).dt.month

        # 根据月份信息将季节进行离散化
        my_data['season'] = my_data['month'].apply(lambda x: (x % 12 + 3) // 3)
        my_data = my_data.query('v100 < 200 and v10 <100')
        my_data['msl_a'] = my_data['msl'] - 100000
        my_data['msl'] = my_data['msl'] / 100
        my_data['lsp'] = my_data['lsp'] * 1000
        my_data['tpe'] = my_data['tpe'] * 1000
        my_data['sp'] = my_data['sp'] / 100
        my_data.loc[:, ['t2m', 'd2m']] = my_data.loc[:, ['t2m', 'd2m']] - 273.15

        return my_data

    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """ 训练

        :return:
        """
        if self.element_model == 'random_forest':
            rf_model = RandomForestClassifier(random_state=42, **self.params)
            self.model = rf_model.fit(X_train, y_train)

        elif self.element_model == 'xgboost':
            xgb_model = XGBClassifier(random_state=42, **self.params)
            self.model = xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=50,
                                       early_stopping_rounds=10)

    def pred(self):
        """
        预测函数
        :return: 
        """
        # 创建ConfigParser对象
        bp = sys.path[0]
        config = configparser.ConfigParser()
        # 读取配置文件
        config.read(f'{os.path.join(bp, "lib", "Thunder", "code","config.ini")}')
        if len(self.dataframe):
            for func in self.methods:
                if func == 'neural_network':
                    path = config.get('Thunder_Model_Path', f'{func}_path')
                    loaded_model = tf.keras.models.load_model(path)
                    # 进行预测
                    predictions = loaded_model.predict(self.dataframe[self.base_fea])

                    # 将预测结果转换为类别标签（假设是二分类问题）
                    predicted_classes = np.argmax(predictions, axis=-1)
                    self.dataframe[func] = predicted_classes
                elif func == 'peiliao':
                    self.dataframe = self.set_storm_level(self.dataframe)

                elif func == 'xgboost' or func == 'random_forest':
                    path = config.get('Thunder_Model_Path', f'{func}_path')
                    th_model = self.load_model(path)
                    self.dataframe[func] = th_model.predict(self.dataframe[self.base_fea])
            if 'ensemble' in self.methods:
                # 应用函数，生成新列
                self.dataframe['ensemble'] = self.dataframe[['xgboost', 'storm', 'random_forest']].apply(self.most_frequent,axis=1)
            else:
                pass
        else:
            raise '输入有误！'

    def get_model(self):
        """
        内部获取模型
        :return: 
        """
        return self.model

    def save_model(self, filename):
        """
        保存模型
        :param filename: "model.pkl" Dtype:Stirng
        :return: None
        """
        my_model = self.get_model()
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(my_model, f)

    @staticmethod
    def load_model(filename):
        """
        加载模型
        :param filename: 
        :return: 
        """
        with open(filename, 'rb') as f:
            new_model = pickle.load(f)
        return new_model

    @staticmethod
    def set_storm_level(df):
        """
        配料法
        :param df: 
        :return: 
        """
        # 创建一个空的Series，用于保存对应行的storm等级
        storm_series = pd.Series(index=df.index, dtype=int)

        # 设置条件1：cape>1000且LI<-2，对应storm等级为1
        condition_1 = (df['cape'] > 100) & (df['LI'] < 5)
        storm_series[condition_1] = 1

        # 设置条件2：cape>2500且LI<-5，对应storm等级为2
        condition_2 = (df['cape'] > 250) & (df['LI'] < 3)
        storm_series[condition_2] = 1

        # 设置其他情况的storm等级为0
        storm_series.fillna(0, inplace=True)

        # 将storm_series添加为新的一列
        df['storm'] = storm_series
        
        return df

    @staticmethod
    # 定义函数，获取一行中出现次数最多的结果
    def most_frequent(row):
        """
        分类集成
        :param row: 
        :return: 
        """
        return row.mode().iloc[0]  # 如果有多个众数，则取第一个


class processer:
    def __init__(self, base_fea, target_fea):
        self.base_fea = base_fea
        self.target_fea = target_fea


# 创建一个更深的神经网络模型
def create_deep_model(input_size, hidden_sizes, num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_size,)))

    for hidden_size in hidden_sizes:
        model.add(tf.keras.layers.Dense(hidden_size, activation='relu'))

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model


# 创建学习率退火函数
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# 自定义数据集类
class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        batch_X = self.X.iloc[batch_indices].values  # 通过 .values 转换为 NumPy 数组
        batch_y = self.y.iloc[batch_indices].values
        return batch_X, batch_y


if __name__ == "__main__":
    base_path = sys.path[0]
    # 创建ConfigParser对象
    config = configparser.ConfigParser()
    # 读取配置文件
    config.read(f'{os.path.join(base_path, "config.ini")}')
    '''
        日志信息包括：（处理订正的时间，出错点）
    '''
    # data = pd.read_csv(
    #     '/mnt/PRESKY/user/wanghaifeng/Projects/046/code/Feature_Projects/td_ds.csv')
    # # xxx_m = XXX_M(factime='2020-01-02', dataframe=dataframe, element='corr-Tem',
    # #               methods=("kalman_filter", "neural_network", "ensemble"))
    # # corr_datarfarame = xxx_m._pred()
    #
    # data.drop_duplicates(subset=['time', 'dtime', 'id', 'time_order'], inplace=True)

    # Thuder_base_fea = ['dtime', 'msl', 'lsp', 'cape', 'sp', 'u10', 'v10', 'tcwv',
    #                    't2m', 'capes', '500t', '850t', '50gh', '500gh', '850gh', '500u',
    #                    '500v', '500r', 'KI', 'SI', 'TT', 'SWEAT']
    # t = Thunder(stime='20230730', order='00', element='Thunder',
    #             methods=("xgboost", "random_forest", "neural_network", 'peiliao', "ensemble"), base_fea=Thuder_base_fea,
    #             target_fea='td',
    #             factime=None, dataframe=None)

    # data['time'] = pd.to_datetime(data['time'])
    # data['stime'] = pd.to_datetime(data['stime'])
    # data['month'] = data['stime'].dt.month
    # data['year'] = data['stime'].dt.year
    # data = data.query('year == 2022')
    # Preded = data.query('stime >= "2022-01-01 00:00:00"')
    #
    # # t.dataframe = t.feature(Preded)
    # # 不做特征工程
    # t.dataframe = Preded
    # print(t.dataframe)
    #
    # pred_td = t.pred()
    # print(pred_td)
    # print(mem.ts(pred_td['td'].values, pred_td['storm'].values))
    # print(mem.ts(pred_td['td'].values, pred_td['xgboost'].values))
    # print(mem.ts(pred_td['td'].values, pred_td['random_forest'].values))
    # print(mem.ts(pred_td['td'].values, pred_td['ensemble'].values))
    # print(mem.ts(pred_td['td'].values, pred_td['neural_network'].values))

    # pred_td.to_csv(
    #     '/share/HPC/home/zhaox_hx/data/aidata/zhangym_tmp/Project/046_project/code/data/预报/corrForest/Thunder/test.csv',
    #     encoding='utf-8')

    # ML_tarin
    # 训练->调参->模型保存
    data = pd.read_csv(
        '/mnt/PRESKY/user/lihaifei/yuchen/46/corr_code/lib/Thunder/code/td_ds.csv')

    # train_e = Thunder(element_model='xgboost', params={
    #
    #     'learning_rate': 0.01,
    #     'max_depth': 10,
    #     'n_estimators': 2000,
    #     'scale_pos_weight': 15,
    #     'gpu_id': 2,
    #     'tree_method': 'gpu_hist',
    #
    # })


    # 这里注释放开 关闭上个类即可更换为rf
    train_e = Thunder(element_model='random_forest', params={
        'n_estimators': 800,
        'class_weight': {0: 0.95, 1: 0.05},
        'max_depth': 8
    })

    new_data = train_e.feature(a=data)
    print(new_data)
    # # 切分数据集
    if 'month' not in new_data.columns:
        new_data['stime'] = pd.to_datetime(new_data['stime'])
        new_data['month'] = new_data['stime'].dt.month
    df = new_data.query('month >=5 and month<10')
    df_1 = df.query('td == 1')
    df_0 = df.query('td == 0')
    df_0_ = df_0.sample(7000)
    df = pd.concat([df_1, df_0_], ignore_index=True)

    Train = df.query('stime < "2021-01-01 00:00:00"')

    Test = df.query('stime >= "2021-01-01 00:00:00" and stime < "2022-01-01 00:00:00"')
    # print(Test)
    base_fea = ['dtime', 'msl', 'lsp', 'cape', 'sp', 'u10', 'v10', 'tcwv',
                't2m', 'capes', '500t', '850t', '50gh', '500gh', '850gh', '500u',
                '500v', '500r', 'KI', 'SI', 'TT', 'SWEAT']
    p_fea = processer(base_fea=base_fea, target_fea='td')

    fea = p_fea.base_fea + [p_fea.target_fea]
    Th_ds = Train[fea]

    # 分离特征列和目标列
    X = Th_ds[p_fea.base_fea]
    y = Th_ds[p_fea.target_fea]
    print(X)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

    train_e.train(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)

    model_path = config.get('Thunder_Model_Path', f'{train_e.element_model}_path')
    print(model_path)
    train_e.save_model(model_path)

    # DL_model_train
    # # 计算类别权重
    # class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # class_weights_dict = {class_idx: weight for class_idx, weight in enumerate(class_weights)}
    #
    # # 创建更深的神经网络模型
    # input_size = X_train.shape[1]
    # hidden_sizes = [64, 32, 16, 8]
    # num_classes = 2
    # batch_size = 128
    # num_epochs = 30
    # model = create_deep_model(input_size, hidden_sizes, num_classes)
    #
    # # 编译模型
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    #               loss='sparse_categorical_crossentropy',
    #               metrics=[tf.keras.metrics.CategoricalAccuracy()])
    #
    # # 创建数据加载器
    # train_dataset = CustomDataset(X_train, y_train, batch_size=batch_size)
    # validation_dataset = CustomDataset(X_valid, y_valid, batch_size=batch_size)
    #
    # # 设置学习率退火回调
    # lr_scheduler = LearningRateScheduler(lr_schedule)
    #
    # # 训练模型
    # model.fit(train_dataset, epochs=num_epochs, validation_data=validation_dataset, callbacks=[lr_scheduler])
    #
    # # 保存模型
    # model.save(config.get('Thunder_Model_Path', 'nn_path'))
    #
    # # 在测试集上评估模型
    # test_loss, test_accuracy = model.evaluate(X_valid, y_valid)
    # print(f"Test Balanced Accuracy: {test_accuracy:.4f}")
