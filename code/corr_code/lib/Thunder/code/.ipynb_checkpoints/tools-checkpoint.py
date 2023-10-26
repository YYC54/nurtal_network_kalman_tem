#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


def save_MaxWins(corresDateFrame,save_corr_path):    
    # 假设你的 DataFrame 名为 df
    # 创建一个列表保存要分割的 sw_max 列名称
    df = corresDateFrame
    sw_max_cols = [1,2]

    # 循环处理每个 sw_max
    for sw_max_col in sw_max_cols:
        # 创建新的 DataFrame
        new_df = df[['time', 'time_order', 'dtime', 'id', 'lon', 'lat', 'stime', '北京时间',
                     f'sw_max{sw_max_col}_pred_nn', f'sw_max{sw_max_col}_pred_kalman', f'sw_max{sw_max_col}_pred_ensemble']].copy()

        # 重命名列名
        new_df.rename(columns={f'sw_max{sw_max_col}_pred_nn': 'sw_max_pred_nn',
                               f'sw_max{sw_max_col}_pred_kalman': 'sw_max_pred_kalman',
                               f'sw_max{sw_max_col}_pred_ensemble': 'sw_max_pred_ensemble'}, inplace=True)

        # 保存 DataFrame 到文件

        new_df.to_csv(save_corr_path.split(".")[0]+f'_{sw_max_col}.csv', index=False)

