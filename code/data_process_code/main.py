# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :root_base
# @File     :main
# @Date     :2023/5/25 18:40
# @Author   :HaiFeng Wang
# @Email    :bigc_HF@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import datetime
import os
import sys
from glob import glob
import pandas as pd
import configparser
from multiprocessing import Pool


def get_path(path, time, order, bounds):
    """
    查找当前日期，当前时效的所有预报文件路径;

    PS: 具体情况需要与现场调整，熟悉glob机制即可

    :param path: 根路径 string
    :param time: 起报时间 string datetime64
    :param order: 00 or 12 string
    :param bounds: [0, 144] array
    :return:file_list list-array
    """
    Y = str(time.year)
    Ym = str(time.year) + str(time.month).zfill(2)
    Ymd = str(time.year) + str(time.month).zfill(2) + str(time.day).zfill(2)
    mdH = str(time.month).zfill(2) + str(time.day).zfill(2) + order
    print(f'{os.path.join(path, Y, Ym, Ymd)}/C1D{mdH}*.bz2')

    file_list = glob(
        f'{os.path.join(path, Y, Ym, Ymd)}/C1D{mdH}*.bz2')
    file_list.sort()
    list_files = []
    for f in file_list:
        file_name = f.split('/')[-1]
        st = datetime.datetime.strptime(
            str(f.split('/')[-4]) + file_name.split('C1D')[-1].split('.bz2')[0][0:6], '%Y%m%d%H')
        et = datetime.datetime.strptime(
            str(f.split('/')[-4]) + file_name.split('C1D')[-1].split('.bz2')[0][8:14], '%Y%m%d%H')
        if bounds[0] <= (et - st).total_seconds() / 3600 <= bounds[1] + 1:
            if (et - st).total_seconds() != 0:
                list_files.append(f)
            else:
                if f[-7:-4] == '011':
                    list_files.append(f)
    return list_files


if __name__ == "__main__":
    base_path = sys.path[0]
    config = configparser.ConfigParser()
    config.read(f'{os.path.join(base_path, "config.ini")}')
    # 解析数据的python环境路径
    myenv_python_path = config.get('DataProcess_Path', 'myenv_python_path')
    # myenv_python_path = '/share/HPC/home/wangxh_hx/soft/anaconda3/envs/myenv/bin/python'

    #原始EC数据存储路径
    data_path = config.get('DataProcess_Path', 'data_path')
    # data_path = '/mnt/PRESKY/data/cmadata/NAFP/ECMF/C1D'

    #处理数据脚本mymain.py 位置
    py_path = config.get('DataProcess_Path', 'py_path')
    # py_path = '/mnt/PRESKY/user/wanghaifeng/product/realtime/mymain.py'

    # start_date = '2017-01-01'
    # end_date = '2019-12-31'
    order = str(sys.argv[1])
    if order == '00':
        dt = datetime.datetime.now()
        print(dt)
        # dt = '2023-09-11 16:00:00.'
        # dt = pd.to_datetime(dt)
    else:
        dt_ = datetime.datetime.now()
        dt = dt_ + pd.Timedelta(-1, unit='D')

    str_dt = datetime.datetime.strftime(dt, '%Y%m%d')
    time = pd.to_datetime(str_dt, format='%Y%m%d')
    path_list = get_path(data_path, time, order, [0, 145])
    path_list.sort()

    # 串行
    for i, f in enumerate(path_list):
        os.system(
            f'{myenv_python_path} {py_path} {f} {i}')

    # 异步并行
    # po = Pool(10)
    # for i, f in enumerate(path_list):
    #     po.apply_async(os.system, args=(f'{myenv_path} {py_path} {f} {i}',))
    # po.close()
    # po.join()
