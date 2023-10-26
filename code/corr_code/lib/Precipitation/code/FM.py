# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
import meteva.method as mem


def freq(df):
    if 'pp6' in df.columns:
        df['bins'] = pd.cut(df['pp6'], bins=[-1, 0.1, 0.25, 1.25, 2.5, 6.25, 8.75, 12.5, 18.75, 25, 37.5, 99999],
                            labels=[0, 0.1, 0.25, 1.25, 2.5, 6.25, 8.75, 12.5, 18.75, 25, 37.5], right=False)
    else:
        df['bins'] = pd.cut(df['tpe6'], bins=[-1, 0.1, 0.25, 1.25, 2.5, 6.25, 8.75, 12.5, 18.75, 25, 37.5, 99999],
                            labels=[0, 0.1, 0.25, 1.25, 2.5, 6.25, 8.75, 12.5, 18.75, 25, 37.5], right=False)
    fre = df['bins'].value_counts() / len(df)
    fre = pd.DataFrame(fre).reset_index()
    fre['time'] = 0
    freq_ = fre.pivot(columns="index", values="bins", index="time")
    return freq_


def freq_match(df, new_df):
    x_interp = new_df['tpe6']
    # 实况概率密度（CDF）
    rt_cdf = np.cumsum(freq(df[['pp6']]).iloc[0])
    # 预报概率密度（CDF）
    pre_cdf = np.cumsum(freq(df[['tpe6']]).iloc[0])
    x = (rt_cdf / pre_cdf).index
    y = (rt_cdf / pre_cdf).values

    # 内插任意 x 对应的任意 y
    #     x_interp = x['tpe']
    y_interp = np.interp(x_interp, x, y)

    # 输出结果
    return y_interp


def th_method(df):
    list_th = []
    for i, th in enumerate(range(1, 20, 1)):
        th_ = th / 10
        df.loc[df['tpe6'] < th, 'tpe6'] = 0
        list_th.append(mem.ts(df['pp6'].values, df['tpe6'].values, grade_list=[0.1]))
    return np.array(list_th).argmax()


def main(tmp, data, window_size):
    # res = np.NAN
    data['frequency_match'] = np.NAN
    for idx, row in data.iterrows():

        now = row
        current_stime = now['stime']
        current_dtime = now['dtime']
        current_to = now['time_order']

        try:

            subset = tmp.query(
                f'stime < "{current_stime}" and stime >= "{pd.to_datetime(current_stime) - pd.Timedelta(window_size, unit="D")}" and dtime=={current_dtime} and time_order == {current_to}')

            if subset.shape[0] == 0:
                now['frequency_match'] = np.NAN
            else:
                th = th_method(subset)
                now['xs'] = freq_match(subset, now)
                now['frequency_match'] = now['xs'] * now['tpe6']
                # 判断条件，将xs小于2的值赋值为999
                if now['frequency_match'] < th:
                    now['frequency_match'] = 0.0
                data.at[idx, 'frequency_match'] = now['frequency_match']

        except KeyError:
            # 当前时刻或6小时前时刻不在数据框中，填充默认值（例如0）
            now['frequency_match'] = np.NAN
            data.at[idx, 'frequency_match'] = now['frequency_match']
    return data
    # data.to_csv('/public/home/zhaox_hx/data/aidata/zhangym_tmp/DATA/ML_dataset/046_DataSet/test_pp.csv', index=False)


if __name__ == "__main__":
    da_xc_all = pd.read_csv('/public/home/zhaox_hx/data/aidata/zhangym_tmp/DATA/ML_dataset/046_DataSet/pp.csv')
    tmp = da_xc_all[['time', 'dtime', 'stime', 'time_order', 'tpe', 'p6', '前6小时累计降水']]
    tmp['time'] = pd.to_datetime(tmp['time'])
    tmp['year'] = tmp['time'].dt.year
    tmp['month'] = tmp['time'].dt.month
    tmp['day'] = tmp['time'].dt.day
    tmp['hour'] = tmp['time'].dt.hour
    tmp_ = tmp.query('stime > "2022-01-01 00:00:00"')
    tmp_['pre_tp_6'] = np.NAN
    main(data=tmp_, window_size=30)