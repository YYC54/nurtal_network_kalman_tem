'''Compute optimal percentile from ensemble member forecasts.

Author: xgz
Update time: 2023-05-23 13:55:53.
'''

# --------Import modules-------------------------
import os
import pickle
# import logging
import numpy as np
from scipy.stats import beta

from lib.Precipitation.code.xb.util_dataset import get_fcst_by_time, load_data
# from util_dataset import get_fcst_by_time, load_data

# import utils

# LOGGER = utils.get_logger('main', logging.DEBUG)


LABEL_NAME_DICT = {
    'ts': r'TS评分',
    'p0': r'漏报率',
    'far': r'空报率',
    'bias': r'预报偏差',
    'ets': r'公平成功指数',
}


def get_hd_weights(n, quantile):
    '''Compute weights for quantile estimate using Harrel-Davis method

    Args:
        n (int): number of samples.
        quantile (float): quantile in [0, 1].
    Returns:
        weights (1darray): 1d array, weights.
    '''

    if np.isclose(quantile, 0):
        weights = np.zeros(n)
        weights[0] = 1.
    elif np.isclose(quantile, 1):
        weights = np.zeros(n)
        weights[-1] = 1.
    else:
        a = quantile * (n + 1)
        b = (1 - quantile) * (n + 1)
        pos = np.arange(n)
        weights = beta.cdf((pos + 1) / n, a, b) - beta.cdf(pos / n, a, b)

    return weights


def compute_scores(tp, fp, fn, tn):
    '''Compute TS score, missing rate, false alarm rate and bias from confusion table

    Args:
        tp (ndarray): true positive.
        fp (ndarray): false positive.
        fn (ndarray): false negative.
        tn (ndarray): true negative.
    Returns:
        scores (dict:) scores = {
                        'ts'   : ts,   # TS score
                        'p0'   : p0,   # misisng rate
                        'far'  : far,  # false alarm rate
                        'bias' : bias, # bias
                    }
    '''

    def safe_divide(num, den):
        flag = den != 0
        res = np.zeros(num.shape)
        res[flag] = num[flag] / den[flag]
        return res

    # ts score
    den = tp + fp + fn
    ts = safe_divide(tp, den)

    # missing rate
    den = tp + fn
    p0 = safe_divide(fn, den)

    # false alarm rate
    den = tp + fp
    far = safe_divide(fp, den)

    # bias
    den = tp + fn
    bias = safe_divide(tp + fp, den)

    # ets
    # den = tp + fp + fn + tn
    # r = safe_divide((tp + fp) * (tp + fn), den)
    # ets = tp / (tp + fp + fn - r) * 100
    # ets = safe_divide(tp, tp+fp+fn-r) * 100

    scores = {
        'ts': ts,
        'p0': p0,
        'far': far,
        'bias': bias,
        # 'ets'  : ets,
    }

    return scores


def compute_confusion_table(obs, pred, thres):
    '''Compute confusion table given a rainfall level

    Args:
        obs (ndarray): observed precipitation values, with shape (n, 1), where
            n is the number of observations.
        pred (ndarray): predicted precipitation values, with shape (n, 1) or
            (n, n_ensemble).
        thres (float): rainfall level.
    Returns:
        tp (float or ndarray): true positive. If <pred> is (n, 1) array, result
            is scalar float. If <pred> is (n, n_ensemble) 2d array, result is
            1d array with shape (n_ensemble, 1). Same for <fp>, <fn>, and <tn>.
        fp (float or ndarray): false positive.
        fn (float or ndarray): false negative.
        tn (float or ndarray): true negative.
    '''

    assert thres > 0, 'thres must > 0'

    obs = np.array(obs)
    pred = np.array(pred)

    if obs.shape != pred.shape:
        obs = np.atleast_2d(obs).T
        pred = np.atleast_2d(pred)

    obs_mat = np.where(obs >= thres, 1, 0)  # (n_fcst_time,  1)
    pred_mat = np.where(pred >= thres, 1, 0)  # (n_fcst_time, n_ensemble)

    tp = (obs_mat * pred_mat).sum(0)
    fp = np.where((pred_mat == 1) & (obs_mat == 0), 1, 0).sum(0)
    fn = np.where((pred_mat == 0) & (obs_mat == 1), 1, 0).sum(0)
    tn = np.where((pred_mat == 0) & (obs_mat == 0), 1, 0).sum(0)

    return tp, fp, fn, tn


def compute_percentile_estimates(df, n_percentiles, n_ensembles, min_ensemble,
                                 lead_time):
    '''Compute percentile estimates from ensemble forecasts

    Args:
        df (DataFrame): DataFrame containing forecast data. Assumes these columns:
            + 'time': forecast time,
            + 'stime': forecast init time,
            + 'pred': forecast value,
            + 'obs': observed value,
        n_percentiles (int): number of percentile levels (in the range of 0-100,
            both ends inclusive) to estimate. E.g. 101 gives 101 levels from
            0% to 100%.
        n_ensembles (int): estimated number of ensemble forecast member for each
            time step. If fewer than this, use whatever number is available. If
            fewer than <min_ensemble>, skip.
        min_ensemble (int): minimum number of ensemble members required. If
            fewer than this, do not use data for that time point.
        lead_time (int): forecast lead time, in number of hours.
    Returns:
        obs_all (ndarray): observed precipitation values, with shape (n, 1), n
            is the number of unique forecast times in <df>.
        perc_estimates_all (ndarray): estimated forecast values at all percentile
            levels, with shape (n, <n_percentiles>).
        ensemble (ndarray): estimated forecast values from the ensemble mean,
            with shape (n, 1).
        std_all (ndarray): "standard/control" forecast values, defined as forecast
            from the latest available forecast, with shape (n, 1).
    '''

    df_fcst_times = df.drop_duplicates(['time']).sort_values(by='time')

    # get Harrel Davis quantile estimate weights
    quantiles = np.linspace(0, 1, n_percentiles)
    q_weights = np.array([get_hd_weights(n_ensembles, qq) for qq in quantiles]).T

    print('len(df_fcst_times) = {}'.format(len(df_fcst_times)))

    perc_estimates_all = []  # store estimates of ensemble percentiles
    ensemble_means_all = []  # store estimates of ensemble average
    obs_all = []  # store observation values
    std_all = []  # store "standard/control" forecast

    # loop through time
    for ii, rowii in df_fcst_times.iterrows():

        tii = rowii['time']
        df_sel = get_fcst_by_time(df, tii)

        # drop rows with nan values
        df_sel = df_sel[~df_sel['pred'].isna()]

        if len(df_sel) < min_ensemble:
            print('df_sel.shape', df_sel.shape, 'skip')
            continue

        # get standard forecast
        std_fcst = df_sel[df_sel.dtime == lead_time]
        if len(std_fcst) == 0:
            print('no fcst at lead time. Set std_fcst to nan')
            std_fcst = np.nan
        else:
            std_fcst = std_fcst['pred'].iloc[0]
            # continue

        # if get fewer ensemble members as expected, use whatever is available
        if len(df_sel) != n_ensembles:
            q_w = np.array([get_hd_weights(len(df_sel), qq) for qq in quantiles]).T
        else:
            q_w = q_weights

        # get ensemble members
        ensemble = df_sel['pred'].to_numpy()
        ensemble.sort()

        # get quantile estimates
        q_esti = ensemble.dot(q_w)

        # collect results
        perc_estimates_all.append(q_esti)
        ensemble_means_all.append(ensemble.mean())
        obs_all.append(df_sel['obs'].iloc[0])
        std_all.append(std_fcst)

    perc_estimates_all = np.array(perc_estimates_all)
    ensemble_means_all = np.array(ensemble_means_all)
    obs_all = np.array(obs_all)
    std_all = np.array(std_all)

    return obs_all, perc_estimates_all, ensemble_means_all, std_all


# -------------Main---------------------------------
if __name__ == '__main__':

    from configs import config

    data_file = config['PRECIP_DATA_FILE']
    output_dir = config['OUTPUT_DIR']
    n_percentiles = config['N_PERCENTILES']
    n_ensembles = config['N_ENSEMBLES']
    min_ensemble = config['MIN_ENSEMBLE']
    lead_time = config['LEAD_TIME']
    sel_station_id = config['SEL_STATION_ID']

    rain_levels = config['RAIN_LEVELS']
    fcst_column = config['FCST_COLUMN']

    year1 = 2017
    year2 = 2021

    # -------------------Read in data-------------------
    df = load_data(data_file)

    # select station id
    df = df[df.id == sel_station_id]

    # select year
    df = df[(df.time.dt.year >= year1) & (df.time.dt.year <= year2)]

    obs_all, perc_estimates_all, ensemble_means_all, std_all = compute_percentile_estimates(
        df, n_percentiles, n_ensembles, min_ensemble, lead_time)

    # -------------Loop through rain levels-------------
    rain_levels = list(rain_levels.values())
    opt_percentiles = {}

    for thres in rain_levels:

        thres = thres / (24 / lead_time)

        # compute TS scores
        tp, fp, fn, tn = compute_confusion_table(obs_all, perc_estimates_all, thres)
        tp_em, fp_em, fn_em, tn_em = compute_confusion_table(obs_all, ensemble_means_all, thres)
        tp_std, fp_std, fn_std, tn_std = compute_confusion_table(obs_all, std_all, thres)

        scores = compute_scores(tp, fp, fn, tn)
        scores_em = compute_scores(tp_em, fp_em, fn_em, tn_em)
        scores_std = compute_scores(tp_std, fp_std, fn_std, tn_std)

        # store percentile at optimal TS
        percentiles = np.linspace(0, 1, n_percentiles) * 100
        best_perc = percentiles[np.argmax(scores['ts'])]

        if np.max(scores['ts']) <= 0.01:
            best_perc = None

        opt_percentiles[thres] = best_perc

        # -------------------Plot scores-------------------
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        rcParams['font.family'] = 'SimHei'

        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        figure, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=100, constrained_layout=True)

        xx = percentiles

        for ii, (kk, vv) in enumerate(scores.items()):

            axii = axes.flatten()[ii]

            em_score = scores_em[kk]
            std_score = scores_std[kk]

            # plot percentile scores
            axii.plot(xx, vv, 'b-o', label='百分位')

            # max TS score percentile
            if kk == 'ts':
                max_ts_idx = np.argmax(vv)
                max_ts_perc = xx[max_ts_idx]

                axii.axvline(max_ts_perc, color='k', ls='-', label='最优百分位')
                axii.text(max_ts_perc, np.max(vv) + 0.01, '{}'.format(max_ts_perc),
                          ha='center', va='bottom',
                          bbox={'facecolor': 'orange', 'alpha': 0.5})

            # plot ensemble mean score
            axii.axhline(em_score, color='r', ls='--', label='集合平均')

            # plot latest forecast at lead time
            axii.axhline(std_score, color='r', ls='-', label='确定预报')

            # plot 50% vertical line
            axii.axvline(0.5, color='k', ls='--', label='中位数')

            axii.set_xlabel(r'百分位(%)')
            axii.set_ylabel(LABEL_NAME_DICT[kk])
            axii.grid(True, axis='both')
            axii.legend(loc=0)

        figure.suptitle('阈值={} mm'.format(thres))
        # figure.tight_layout()
        figure.show()

        # ----------------- Save plot------------
        plot_save_name = 'new_percentiles_thres_{:.2f}_lead_time_{}.png'.format(thres, lead_time)
        plot_save_name = os.path.join(output_dir, plot_save_name)
        os.makedirs(output_dir, exist_ok=True)
        print('\n# <cmpt_optimal_percentile>: Save figure to:', plot_save_name)
        figure.savefig(plot_save_name, dpi=100, bbox_inches='tight')

        plt.close(figure)

    output_dir = './data'
    os.makedirs(output_dir, exist_ok=True)
    abpath_out = os.path.join(output_dir, 'optimal_percentiles.pkl')
    print('Saving optimal percentile dict to:', abpath_out)
    pickle.dump(opt_percentiles, open(abpath_out, 'wb'))
