# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :046
# @File     :nn.py
# @Date     :2023/8/21 16:32
# @Author   :HaiFeng Wang
# @Email    :bigc_HF@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[1], True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
class RegressionModel:
    def __init__(self, t_type):
        self.data_frame = None
        self.model = None
        self.scaler = None
        self.type = t_type

    def load_data(self, data_frame):
        if self.type == 'Train':
            data_frame = data_frame[['dtime', 'v100', 'tcc', 'msl',
                                     'lsp', 'cape', 'sst', 'lcc', 'u100', 'skt', 'tpe',
                                     'sp', 'd2m', 't2m', 'p3020', 'fg310', 'tcwv', 'tcw',
                                     'fal', 'fzra', 'capes',
                                     'time_order', 'tpe6', 'pp6']]
        else:
            data_frame = data_frame[['dtime', 'v100', 'tcc', 'msl',
                                     'lsp', 'cape', 'sst', 'lcc', 'u100', 'skt', 'tpe',
                                     'sp', 'd2m', 't2m', 'p3020', 'fg310', 'tcwv', 'tcw',
                                     'fal', 'fzra', 'capes',
                                     'time_order', 'tpe6']]
        self.data_frame = data_frame

    def preprocess_data(self):
        # 提取特征和标签
        X = self.data_frame.drop(columns=['pp6']).values
        y = self.data_frame['pp6'].values

        # self.scaler = StandardScaler()
        # X = self.scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
            tf.keras.layers.Dropout(0.2),  # 添加Dropout层进行正则化
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),  # 添加Dropout层进行正则化
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)  # 输出层无需激活函数
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, num_epochs=30, batch_size=32):
        self.model.fit(self.X_train, self.y_train, epochs=num_epochs, batch_size=batch_size, verbose=2)

    def evaluate_model(self):
        test_loss = self.model.evaluate(self.X_test, self.y_test)
        print(f'Test Loss: {test_loss:.4f}')

    def predict(self, new_data_frame):
        self.load_data(new_data_frame)
        self.data_frame['dtime'] = self.data_frame['dtime'].astype('float64')
        self.data_frame['time_order'] = self.data_frame['time_order'].astype('float64')
        if 'pp6' in self.data_frame.columns:

            new_data = self.data_frame.drop(columns=['pp6']).values
        else:
            new_data = self.data_frame.values
        # new_data_scaled = self.scaler.transform(new_data)
        tensor = tf.convert_to_tensor(new_data)
        predictions = self.model.predict(tensor)
        return predictions

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)


if __name__ == "__main__":
    # 创建模型对象
    model = RegressionModel('Train')

    # 加载数据到DataFrame
    mydata = pd.read_csv('/mnt/PRESKY/user/lihaifei/yuchen/46/corr_code/lib/Precipitation/code/xb/pp.csv')

    mydata.rename(columns={'前6小时累计降水': 'pp6', 'p6': 'tpe6'}, inplace=True)
    mydata.drop(['level', 'stime', 'time', 'storm', 'ct',
                 'wd', 'ws', 'wp', 'dp', 'ps', 'rh', 'pp', 'td', 'sw_max',
                 'sw_avg', 'vis', 'sw_max1', 'sw_avg1', 'sw_max2', 'sw_avg2', 'dt'],
                axis=1, inplace=True)
    mydata.dropna(inplace=True)
    a = mydata.query('pp6 > 0.1')
    b = mydata.query('pp6 <= 0.1').sample(frac=0.2)
    c = pd.concat([a, b], ignore_index=True)
    del mydata

    model.load_data(c)

    # 数据预处理
    model.preprocess_data()

    # 构建模型
    model.build_model()

    # 训练模型
    model.train_model()

    # 评估模型
    model.evaluate_model()

    # 保存模型
    model.save_model('regression_model.h5')
    '''
    # 加载模型
    loaded_model = RegressionModel()
    loaded_model.load_model('regression_model.h5')
    exit()
    # 使用加载的模型进行预测
    # new_data = pd.DataFrame({'feature1': [...], 'feature2': [...], ...})  # 替换为实际的新数据
    # predictions = loaded_model.predict(new_data)
    # print(predictions)
    '''
