# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :046
# @File     :makePredata.py
# @Date     :2023/8/17 9:46
# @Author   :HaiFeng Wang
# @Email    :bigc_HF@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import pandas as pd
import numpy as np
import psycopg2
import datetime
import os
import sys
import configparser


def conn_ht_data_center(conner, table_name='xichangsk_history', time=None, window=None):
    """
    '221.122.67.146' #windowss
    '10.1.10.124' # Linux
    '11.175.24.249' # 甲方HJJ
    """
    cursor = conner.cursor()

    # 获取全部字段
    cursor.execute(
        f"select string_agg(column_name,',') from information_schema.columns where table_schema='public' and table_name='{table_name}'")  # 执行SQL语句
    columns_names = cursor.fetchall()
    cursor.execute(
        f"SELECT * FROM {table_name} where dt between '{time + datetime.timedelta(days=-window - 1)}' and '{time}'")

    rows = cursor.fetchall()
    columns_name = list(columns_names[0])[0].split(',')

    dataframe = pd.DataFrame(rows, columns=columns_name)
    dataframe['dt'] = pd.to_datetime(dataframe['dt'])
    dataframe['dt'] = dataframe['dt'].apply(lambda x: x - pd.Timedelta(8, unit='H'))
    cursor.close()
    # conner.close()
    return dataframe


def rt2p6(rt_data):

    rt_data['dt'] = pd.to_datetime(rt_data['dt'])
    rt_data['year'] = rt_data['dt'].dt.year
    rt_data['month'] = rt_data['dt'].dt.month
    rt_data['day'] = rt_data['dt'].dt.day
    rt_data['hour'] = rt_data['dt'].dt.hour
    tl = [x for x in range(0, 24) if x % 6 == 0]

    a = pd.cut(rt_data['hour'], bins=tl, labels=[6, 12, 18])
    rt_data['new'] = [x for x in a]
    rt_data['new'] = rt_data['new'].fillna(0)
    print(rt_data)
    ndata_nor = rt_data.query('hour != 0')
    ndata_zero = rt_data.query('hour == 0')
    ndata_zero['dt'] = ndata_zero['dt'].apply(lambda x: x + pd.Timedelta(1, unit='D'))
    ndata_zero['time'] = ndata_zero['dt']
    nor_pp = ndata_nor.groupby(['year', 'month', 'day', 'new'])['pp'].sum().reset_index()
    nor_pp['time'] = nor_pp['year'].astype('string') + '-' + nor_pp['month'].astype('string') + '-' + nor_pp[
        'day'].astype('string')
    print(nor_pp.head())
    nor_pp['time'] = pd.to_datetime(nor_pp['time'], format='%Y-%m-%d')
    nor_pp['time'] = nor_pp.apply(lambda x: x['time'] + pd.Timedelta(x['new'], unit='H'), axis=1)
    nor_pp['time'] = nor_pp.apply(lambda x: x['time'] + pd.Timedelta(1, unit='D') if x['new'] == 0 else x['time'],
                                  axis=1)
    pp_a = ndata_zero[['time', 'pp']]
    pp_b = nor_pp[['time', 'pp']]
    pp_rt = pd.concat([pp_a, pp_b], ignore_index=True)
    pp_rt = pp_rt.groupby(['time'])['pp'].sum().reset_index()
    pp_rt.rename(columns={'pp': 'pp6'}, inplace=True)

    return pp_rt


def pred2p6(ndata, mode_ec, stime): 
    if 'time_order' not in mode_ec.columns:
        mode_ec['time_order'] = mode_ec['stime'].dt.hour
    if 'tpe' not in mode_ec.columns:
        mode_ec.rename(columns={'tp': 'tpe'}, inplace=True)
    mode_ec.drop_duplicates(subset=['stime', 'dtime', 'time_order'], inplace=True)

    for idx, row in mode_ec.iterrows():
        try:
            now = row
            current_stime = now['stime']
            current_dtime = now['dtime']
            current_to = now['time_order']

            subset = mode_ec.query(
                f'stime == "{current_stime}" and dtime=={current_dtime - 6} and time_order == {current_to}')
            if subset.shape[0] == 0:
                if current_dtime <= 6:
                    mode_ec.at[idx, 'tpe6'] = now['tpe']
                else:
                    mode_ec.at[idx, 'tpe6'] = np.NAN
            else:
                now_tpe6 = now['tpe'] - subset['tpe']
                mode_ec.at[idx, 'tpe6'] = now_tpe6
        except Exception:
            mode_ec.at[idx, 'tpe6'] = np.NAN

    # mode_pp = mode_ec[['stime', 'dtime', 'time', 'tpe6']]
    # 将满足条件的行的'B'列设置为0
    mode_ec.loc[mode_ec['dtime'] == 0, 'tpe6'] = 0
    mode_ec['tpe6'] = mode_ec['tpe6'] * 1000
    dtime_list = [x for x in range(0, 121) if x % 6 == 0]
    mode_pp_pred = mode_ec.query(f'stime == "{stime}" and dtime in {dtime_list}')
    pp_dataset_subset = pd.merge(mode_ec, ndata, on=['time'])
    mode_pp_pred = mode_pp_pred.sort_values(by=['dtime'])
    return pp_dataset_subset, mode_pp_pred


if __name__ == "__main__":
    windows_size = 30
    real_time = datetime.datetime(2017, 6, 26, 0)
    t = real_time + datetime.timedelta(days=-1)

    start_time = t + datetime.timedelta(days=-windows_size)
    end_time = t

    config = configparser.ConfigParser()

    config.read(r'D:\046\other_code\lib\Thunder\code\config.ini')

    db_ip = config.get('Database', 'ip')
    db_post = config.get('Database', 'post')
    db_user = config.get('Database', 'user')
    db_password = config.get('Database', 'password')
    db_database_name = config.get('Database', 'database_name')
    db_table = config.get('Database', 'table')

    conn = psycopg2.connect(database=db_database_name, user=db_user,
                            password=db_password, host=db_ip, port=db_post)
    df_sta = conn_ht_data_center(conner=conn, table_name=db_table, time=t + datetime.timedelta(hours=-16),
                                 window=windows_size)

    mode_ec = pd.read_csv(r'D:\046\other_code\lib\Tem\dataset\qiancengfeng_leibao.csv')

    mode_ec['tpe6'] = np.NAN
    mode_ec = mode_ec.query(f'"{start_time}" <= stime <= "{real_time}"')
    mode_ec['stime'] = pd.to_datetime(mode_ec['stime'])
    mode_ec['time'] = pd.to_datetime(mode_ec['time'])

    ndata = rt2p6(df_sta)
    data, pred = pred2p6(ndata, mode_ec, real_time)
