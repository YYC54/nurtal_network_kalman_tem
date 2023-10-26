'''Adjust forecast using optimal percentiles.

Author: xgz
Update time: 2023-05-24 13:30:20.
'''

# --------Import modules-------------------------
import os
# import logging
import pickle
import datetime
import numpy as np
# import pandas as pd
from lib.Precipitation.code.xb.util_dataset import get_fcst_by_time, load_data
from lib.Precipitation.code.xb.cmpt_optimal_percentile import get_hd_weights, compute_confusion_table, compute_scores

# from util_dataset import get_fcst_by_time, load_data
# from cmpt_optimal_percentile import get_hd_weights, compute_confusion_table, compute_scores

# import utils

# LOGGER = utils.get_logger('main', logging.DEBUG)

LABEL_NAME_DICT = {
    'ts': r'TS评分',
    'p0': r'漏报率',
    'far': r'空报率',
    'bias': r'预报偏差',
    'ets': r'公平成功指数',
}


def adjust_to_opt_percentile(df, opt_percentiles):
    """Adjust precipitation forecasts using optimal percentile of ensemble members

    Args:
        df (DataFrame): DataFrame containing forecast data. Assumes these columns:
            + 'time': forecast time,
            + 'stime': forecast init time,
            + 'pred': forecast value,
            + 'obs': observed value,
        opt_percentiles (dict): keys: rain level in mm, values: optimal percentile level.
    Returns:
        obs_all (ndarray): observed precipitation values, with shape (n, 1), n
            is the number of unique forecast times in <df>.
        perc_estimates_all (ndarray): estimated forecast values at all percentile
            levels, with shape (n, <n_percentiles>).
        ensemble (ndarray): estimated forecast values from the ensemble mean,
            with shape (n, 1).
        res_df (DataFrame): <df> with additional column 'tp_24_adj', adjusted forecast.
    """

    df_fcst_times = df.drop_duplicates(['time']).sort_values(by='time')

    # order rain levels from highest to lowest
    rain_levels = list(opt_percentiles.keys())
    rain_levels.sort(reverse=True)

    # perc_estimates_all = []
    # ensemble_means_all = []
    # obs_all = []

    res_df = df.copy()
    res_df.loc[:, 'pred_adj'] = res_df['pred']

    # loop through time
    for ii, rowii in df_fcst_times.iterrows():

        tii = rowii['time']
        df_sel = get_fcst_by_time(df, tii)

        # drop rows with nan values
        df_sel = df_sel[~df_sel['pred'].isna()]

        if len(df_sel) == 0:
            continue

        # get ensemble members
        ensemble = df_sel['pred'].to_numpy()
        ensemble.sort()

        # get optimal quantile estimate
        opt_esti = 0.

        for kk in rain_levels:
            opt_perc = opt_percentiles[kk]

            if opt_perc is None:
                continue

            opt_perc /= 100
            opt_weight = get_hd_weights(len(df_sel), opt_perc)
            tmp = ensemble.dot(opt_weight)

            if tmp >= kk:
                opt_esti = tmp
                break

        # perc_estimates_all.append(opt_esti)
        # ensemble_means_all.append(ensemble.mean())
        # obs_all.append(df_sel['obs'].iloc[0])

        res_df.loc[res_df.time == tii, 'pred_adj'] = opt_esti

    # perc_estimates_all = np.array(perc_estimates_all)
    # ensemble_means_all = np.array(ensemble_means_all)
    # obs_all = np.array(obs_all)

    # return obs_all, perc_estimates_all, ensemble_means_all, res_df
    return res_df


def post_process(df):
    utc_time = df['time']
    local_time = utc_time + datetime.timedelta(hours=8)

    df.loc[:, '北京时间'] = local_time
    # df = df.rename(columns={'obs': 'tp24_true', 'tp_24': 'tp24_EC'})
    df = df.rename(columns={'obs': 'p6_true', 'p6': 'p6_EC'})

    return df


# -------------Main---------------------------------
if __name__ == '__main__':

    from configs import config

    data_file = config['PRECIP_DATA_FILE']
    plot_dir = config['PLOT_DIR']
    output_dir = config['OUTPUT_DIR']
    n_percentiles = config['N_PERCENTILES']
    n_ensembles = config['N_ENSEMBLES']
    min_ensemble = config['MIN_ENSEMBLE']
    lead_time = config['LEAD_TIME']
    sel_station_id = config['SEL_STATION_ID']
    rain_levels = config['RAIN_LEVELS']

    thres_levels = list(rain_levels.values())
    thres_levels = [x / (24 / lead_time) for x in thres_levels]

    year1 = 2022
    year2 = 2023

    # -------------Load optimal percentiles-------------
    abspath = os.path.join('data', 'optimal_percentiles.pkl')
    optimal_percentiles = pickle.load(open(abspath, 'rb'))

    # -------------------Read in data-------------------
    df = load_data(data_file)

    # select station id
    df = df[df.id == sel_station_id]

    # select year
    df = df[(df.time.dt.year >= year1) & (df.time.dt.year < year2)]

    obs_all, perc_estimates_all, ensemble_means_all, df_adj = adjust_to_opt_percentile(
        df, optimal_percentiles)

    # -----------Save adjusted results to csv-----------
    df_adj = post_process(df_adj)

    file_out_name = 'Precipitation_percentile_{}.csv'.format(year1)
    abpath_out = os.path.join(output_dir, file_out_name)
    print('Saving output to {}'.format(abpath_out))
    df_adj.to_csv(abpath_out, index=False, encoding='utf-8_sig')

    # compute TS scores
    scores_all = {}
    scores_em_all = {}

    for thres in thres_levels:
        print('computing scores for thres', thres)

        tp, fp, fn, tn = compute_confusion_table(obs_all, perc_estimates_all, thres)
        tp_em, fp_em, fn_em, tn_em = compute_confusion_table(obs_all, ensemble_means_all, thres)
        scores = compute_scores(tp, fp, fn, tn)
        scores_em = compute_scores(tp_em, fp_em, fn_em, tn_em)

        scores_all[thres] = scores
        scores_em_all[thres] = scores_em

    # # -------------------Plot scores-------------------
    # import matplotlib.pyplot as plt
    #
    # figure, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=100, constrained_layout=True)
    #
    # xx = np.arange(len(scores_all))
    # width = 0.2
    #
    # for ii, (kk, vv) in enumerate(scores.items()):
    #     axii = axes.flatten()[ii]
    #
    #     opt_scores = [scores_all[tt][kk] for tt in thres_levels]
    #     axii.bar(xx - width, opt_scores, width=width, color='orange', label='最优百分位')
    #
    #     em_scores = [scores_em_all[tt][kk] for tt in thres_levels]
    #     axii.bar(xx, em_scores, width=width, color='c', label='集合平均')
    #
    #     axii.set_xticks(xx)
    #     axii.set_xticklabels(thres_levels)
    #     axii.set_xlabel('降雨等级 (mm/{} hr)'.format(24 / lead_time))
    #
    #     axii.set_ylabel(LABEL_NAME_DICT[kk])
    #     axii.grid(True, axis='both')
    #     axii.legend(loc=0)
    #
    # figure.show()

    # # ----------------- Save plot------------
    # plot_save_name = 'new_adjusted_percentiles_lead_time_{}.png'.format(lead_time)
    # plot_save_name = os.path.join(plot_dir, plot_save_name)
    # os.makedirs(plot_dir, exist_ok=True)
    # print('\n# <cmpt_optimal_quantile>: Save figure to:', plot_save_name)
    # figure.savefig(plot_save_name, dpi=100, bbox_inches='tight')
    #
    # plt.close(figure)
