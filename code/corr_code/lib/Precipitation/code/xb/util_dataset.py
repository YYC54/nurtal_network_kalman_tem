'''Load precipitation data from station and ensemble forecasts

Author: xgz
Update time: 2023-05-22 14:31:27.
'''

# --------Import modules-------------------------
import datetime
import dateutil
import pandas as pd

# import utils

# LOGGER = utils.get_logger('main', logging.DEBUG)


class DataError(Exception):
    def __init__(self, msg=None):
        Exception.__init__(self)
        self._msg = str(msg)

    def __str__(self):
        return self.message


def str_to_datetime(text, format=None):
    '''Convert string time stamp to datetime object, fuzzy format parsing version'''

    if format is None:
        try:
            res = dateutil.parser.parse(text, fuzzy=True)
        except:
            raise DataError('text {} failed to parse to datetime'.format(text))
    else:
        res = datetime.datetime.strptime(text, format)

    return res


def utc_to_local(utc_time):
    return utc_time + datetime.timedelta(hours=8)


def local_to_utc(local_time):
    return local_time - datetime.timedelta(hours=8)


def get_fcst_by_time(df, fcst_time_utc):
    df_sel = df[df.time == fcst_time_utc]
    df_sel = df_sel.sort_values(by='stime')
    df_sel.reset_index(drop=True, inplace=True)

    # remove rows where fcst_init_time == fcst_time
    df_sel = df_sel[~(df_sel.time == df_sel.stime)]

    '''
    # fill NaN with 2* 12hr tp
    #nan_idx = df_sel['tp_24'].index[df_sel['tp_24'].apply(np.isnan)]
    nan_idx = df_sel.loc[pd.isna(df_sel['tp_24']), :].index

    for ii in nan_idx:
        repii = df_sel.iloc[ii].tp * 2
        df_sel.loc[ii, 'tp_24'] = repii
    '''

    return df_sel


def load_data(abspath):
    '''Read in precipitation forecast and observation data and preprocess

    Args:
        abspath (str): absolute path to csv data file.
    Returns:
        df (DataFrame): precipitation data.

    Assumes these columns are present in the data file:

         ['time', 'dtime', 'tp_24', 'stime', 'pp', 'id', 'lon', 'lat']

    Do some preprocessing:

        + change 'time' and 'stime' columns to datetime objects.
        + rename 'pp' to 'obs'.
    '''

    df = pd.read_csv(abspath, encoding='utf-8',
                     # usecols=['time', 'dtime', 'tp_24', 'stime', 'pp',
                     usecols=['time', 'dtime', 'p6', 'stime', '前6小时累计降水',
                              'id', 'lon', 'lat'])

    df.loc[:, 'time'] = pd.to_datetime(df['time'])
    df.loc[:, 'stime'] = pd.to_datetime(df['stime'])
    # df = df.rename(columns={'pp': 'obs', 'tp_24': 'tp_24'})
    df = df.rename(columns={'前6小时累计降水': 'obs', 'p6': 'pred'})
    df = df.drop_duplicates(['time', 'id', 'stime'])
    obs = df['obs'].clip(0, None)
    df.loc[:, 'obs'] = obs
    pred = df['pred'].clip(0, None)
    df.loc[:, 'pred'] = pred

    return df


# -------------Main---------------------------------
if __name__ == '__main__':
    from configs import config

    data_file = config['PRECIP_DATA_FILE']

    # -------------------Read in data-------------------
    df = pd.read_csv(data_file, encoding='utf-8',
                     usecols=['time', 'dtime', 'tp', 'tp_24', 'stime', '降水量',
                              'id', 'lon', 'lat',
                              ])

    df.loc[:, 'time'] = pd.to_datetime(df['time'])
    df.loc[:, 'stime'] = pd.to_datetime(df['stime'])
    df = df.rename(columns={'降水量': 'obs'})
