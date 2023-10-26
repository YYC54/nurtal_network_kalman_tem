# -*- coding: utf-8-sig -*-
"""
-------------------------------------------------
# @Project  :root_base
# @File     :mymain
# @Date     :2023/5/23 11:30
# @Author   :HaiFeng Wang
# @Email    :bigc_HF@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import bz2
import numpy as np
import pandas as pd
import xarray as xr
import os
# import multiprocessing
from netCDF4 import *
import datetime
import meteva.base as meb
import sys
import configparser

def check_path(_path):
    """ Check weather the _path exists. If not, make the dir. """
    # 判断路径的文件夹是否存在（最后一个斜杠前的文件夹），如果不存在，则创建该路径
    if os.path.dirname(_path):
        if not os.path.exists(os.path.dirname(_path)):
            os.makedirs(os.path.dirname(_path))


def my_bz2(bz2file, npath):
    """将bz2数据解压到对应时间文件夹下

    :param bz2file:
    :return:
    """
    print(f'对{bz2file}进行解压。')
    try:
        zipfile = bz2.BZ2File(bz2file)  # open the file
        data = zipfile.read()  # get the decompressed data
        open(npath, 'wb').write(data)  # write a uncompressed file
    except Exception as e:
        print(e)


def process(p, tp):
    """

    :param p:
    :param tp:
    :return:
    """
    workdir_path = '/mnt/PRESKY/user/lihaifei/yuchen/46/data_process_code'
    name = p.split('/')[-1].split('_')[-1]
    Y_ = p.split('/')[-4]
    Ym_ = p.split('/')[-3]

    stime = Y_ + name[3:9]
    ettime = Y_ + name[-13:-7]
    new_name = f'C1D_EC_{stime}_{ettime}'
    new_pos = os.path.join(tp, Y_, Ym_, new_name)

    hour_delta = (pd.to_datetime(ettime, format='%Y%m%d%H') - pd.to_datetime(stime,
                                                                             format='%Y%m%d%H')).total_seconds() / 3600
    ymd = pd.to_datetime(str(stime), format='%Y%m%d%H').strftime('%Y%m%d')
    try:
        check_path(new_pos)
        my_bz2(p, new_pos)
        if not os.path.exists(os.path.join(tp, 'rule.filter')):
            os.system(f'cp -r {workdir_path}/rule.filter {tp}')
        os.system(f'cd {tp} ; grib_filter rule.filter {new_pos}')
    except Exception as e:
        print(e)
        p1 = f'{tp}/ecmf_{ymd}_{int(hour_delta)}_sfc.grib1'
        p2 = f'{tp}/ecmf_{ymd}_{int(hour_delta)}_pl.grib1'
        p3 = f'{tp}/ecmf_{ymd}_{int(hour_delta)}_1.grib2'
        p4 = f'{tp}/ecmf_{ymd}_{int(hour_delta)}_105.grib2'
        print(f"{new_pos} has been processed; but have error")
        return new_pos, p1, p2, p3, p4, stime, ettime, int(hour_delta)

    p1 = f'{tp}/ecmf_{ymd}_{int(hour_delta)}_sfc.grib1'
    p2 = f'{tp}/ecmf_{ymd}_{int(hour_delta)}_pl.grib1'
    p3 = f'{tp}/ecmf_{ymd}_{int(hour_delta)}_1.grib2'
    p4 = f'{tp}/ecmf_{ymd}_{int(hour_delta)}_105.grib2'
    print(f"{new_pos} has been processed.")
    return new_pos, p1, p2, p3, p4, stime, ettime, int(hour_delta)


def main(df, num, file_path, target_path):
    sta = meb.sta_data(df)
    sta_base = sta

    print(f'当前文件为第{num}个：{file_path}')
    start_time = datetime.datetime.now()
    print(f'当前时间为：{start_time}')

    new_path, sfc, pl, ptype, lnsp, st, et, dt_hour = process(file_path, target_path)
    # print(st)

    try:
        filename = new_path.split('/')[-1]
        ds_pl = xr.open_dataset(f'{pl}', engine='cfgrib', backend_kwargs={"filter_by_keys": {'edition': 1}})
        ds_sfc = xr.open_dataset(f'{sfc}', engine='cfgrib',
                                 backend_kwargs={"filter_by_keys": {'edition': 1}})

        print('对', {file_path}, '进行插值')
        # sfc
        sfc_key_list = ['v100', 'tcc', 'sf', 'msl', 'lsp', 'cape', 'sst', 'lcc', 'sd', 'cp', 'u100', 'skt',
                        'tp', 'sp', 'd2m', 'u10', 'v10', 't2m', 'p3020', 'fg310', 'tcwv', 'rsn', 'deg0l',
                        'mn2t3', 'mx2t3', 'tcw', 'fal', 'fzra', 'capes']
        sfc_sta = np.NAN
        for idx1, k1 in enumerate(sfc_key_list):

            try:
                da = ds_sfc[f'{k1}']
                da = da.rename({'latitude': 'lat'})
                da = da.rename({'longitude': 'lon'})
                grd = meb.xarray_to_griddata(da)
                meb.set_griddata_coords(grd, name=f'{k1}', member_list=[f"{k1}"])
                # 设置站点表中缺失的属性

                meb.set_stadata_coords(sta_base, level=0, time=et, dtime=dt_hour)
                station = meb.interp_gs_linear(grd, sta_base)
                if idx1 == 0:
                    sfc_sta = station
                else:
                    sfc_sta[f'{k1}'] = station[f'{k1}']
            except KeyError:
                print(k1)
                if idx1 == 0:
                    sfc_sta = sta_base
                else:
                    sfc_sta[f'{k1}'] = np.NAN
                    continue

        sfc_sta['stime'] = pd.to_datetime(st, format='%Y%m%d%H')
        sfc_sta['time'] = pd.to_datetime(st, format='%Y%m%d%H') + pd.Timedelta(dt_hour, unit='H')
        sfc_sta['dtime'] = dt_hour

        # pl
        pl_key_list = ['t', 'gh', 'u', 'v', 'r', 'w', 'q', 'd', 'pv']
        pl_sta = np.NAN
        real_index = 0
        for idx2, k2 in enumerate(pl_key_list):
            try:
                da = ds_pl[f'{k2}']
                pre_level = da['isobaricInhPa']
                for l in pre_level:
                    nda = da.sel(isobaricInhPa=l)
                    grd = meb.xarray_to_griddata(nda)
                    meb.set_griddata_coords(grd, name=f'{k2}', member_list=[f"{k2}"])
                    # 设置站点表中缺失的属性
                    meb.set_stadata_coords(sta_base, level=0, time=et, dtime=dt_hour)
                    station = meb.interp_gs_nearest(grd, sta_base)
                    real_index += 1
                    if real_index == 1:
                        station.rename(columns={f'{k2}': f'{int(l)}{k2}'}, inplace=True)
                        pl_sta = station
                    else:
                        station.rename(columns={f'{k2}': f'{int(l)}{k2}'}, inplace=True)
                        pl_sta[f'{int(l)}{k2}'] = station[f'{int(l)}{k2}']

            except KeyError:
                print(k2)
                if idx2 == 0:
                    pl_sta = sta_base
                else:
                    pl_sta[f'{k2}'] = np.NAN
                    continue

        pl_sta['stime'] = pd.to_datetime(st, format='%Y%m%d%H')
        pl_sta['time'] = pd.to_datetime(st, format='%Y%m%d%H') + pd.Timedelta(dt_hour, unit='H')
        pl_sta['dtime'] = dt_hour

        res_sta = pd.merge(sfc_sta, pl_sta, on=['level', 'stime', 'time', 'dtime', 'id', 'lon', 'lat'])
        # sfc_2
        if os.path.exists(ptype):
            ds_sfc_2 = xr.open_dataset(f'{ptype}', engine='cfgrib',
                                       backend_kwargs={"filter_by_keys": {'edition': 2}})

            da = ds_sfc_2['ptype']
            da = da.rename({'latitude': 'lat'})
            da = da.rename({'longitude': 'lon'})
            grd = meb.xarray_to_griddata(da)
            meb.set_griddata_coords(grd, name=f'ptype', member_list=[f"ptype"])
            # 设置站点表中缺失的属性

            meb.set_stadata_coords(sta_base, level=0, time=et, dtime=dt_hour)
            station = meb.interp_gs_nearest(grd, sta_base)
            res_sta['ptype'] = station['ptype']
        else:
            res_sta['ptype'] = np.NAN

        if os.path.exists(lnsp):
            ds_sfc_2 = xr.open_dataset(f'{lnsp}', engine='cfgrib',
                                       backend_kwargs={"filter_by_keys": {'edition': 2}})

            da = ds_sfc_2['lnsp']
            da = da.rename({'latitude': 'lat'})
            da = da.rename({'longitude': 'lon'})
            grd = meb.xarray_to_griddata(da)
            meb.set_griddata_coords(grd, name=f'lnsp', member_list=[f"lnsp"])
            # 设置站点表中缺失的属性

            meb.set_stadata_coords(sta_base, level=0, time=et, dtime=dt_hour)
            station = meb.interp_gs_nearest(grd, sta_base)
            res_sta['lnsp'] = station['lnsp']
        else:
            res_sta['lnsp'] = np.NAN

        end_time = datetime.datetime.now()
        print(f'处理完成当前时间为：{end_time}')
        print(f'单个文件总耗时：{(end_time - start_time).total_seconds()} seconds')
        # print(file_path)
        # print(file_path.split("/")[6])
        save_path = f'{os.path.join(save_base_path, str(pd.to_datetime(st, format="%Y%m%d%H").year), str(pd.to_datetime(st, format="%Y%m%d%H").year) + str(pd.to_datetime(st, format="%Y%m%d%H").month).zfill(2), str(pd.to_datetime(st, format="%Y%m%d%H").year) + str(pd.to_datetime(st, format="%Y%m%d%H").month).zfill(2) + str(pd.to_datetime(st, format="%Y%m%d%H").day).zfill(2), file_path.split("/")[-1][0:9] + ".csv")}'

        check_path(save_path)

        res_sta.to_csv(save_path, mode="a", index=False, header=not os.path.exists(save_path), encoding='utf-8-sig')

        # print('res_sta columns:')
        # print(res_sta.columns.values)
        p_all = sfc.replace('_sfc.grib1', '*')
        os.system(f'rm -rf {p_all}')
        os.system(f'rm -rf {new_path}')
    except FileNotFoundError:
        p_all = sfc.replace('_sfc.grib1', '*')
        os.system(f'rm -rf {p_all}')
        os.system(f'rm -rf {new_path}')
        print(FileNotFoundError)


if __name__ == "__main__":
    base_path = sys.path[0]
    file_path = sys.argv[1]
    num = sys.argv[2]
    config = configparser.ConfigParser()
    config.read(f'{os.path.join(base_path, "config.ini")}')

    # Bz2解压缩后的文件路径
    # target_path = '/share/HPC/home/zhaox_hx/data/aidata/zhangym_tmp/DATA/ML_dataset/Corr_datasets/tmp'
    target_path = config.get('DataProcess_Path', 'target_path')

    # 插值后的结果存储路径
    # save_base_path = '/share/HPC/home/zhaox_hx/data/aidata/zhangym_tmp/DATA/ML_dataset/Corr_datasets'
    save_base_path = config.get('DataProcess_Path', 'save_base_path')

    data = {
        'id': [4, 2, 15, 3, 8, 1, 10, 838961],
        "lon": [102.09, 102.03, 102.20, 102.21, 102.16, 102.25, 102.22, 102.18],
        "lat": [28.19, 28.23, 28.03, 27.91, 28.19, 27.90, 27.88, 27.95]
    }
    df = pd.DataFrame(data)

    # if os.path.exists('/mnt/PRESKY/user/wanghaifeng/product/realtime/Country_sta.csv'):
    #     df_c = pd.read_csv('/mnt/PRESKY/user/wanghaifeng/product/realtime/Country_sta.csv')
    #     df_c_ = df_c[['id', 'lon', 'lat']]
    #     df = pd.concat([df, df_c_], ignore_index=True)

    main(df=df, num=num, file_path=file_path, target_path=target_path)
