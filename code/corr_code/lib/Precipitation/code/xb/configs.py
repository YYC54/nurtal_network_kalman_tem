config = {

    'LOG_DIR': './logs',
    'PLOT_DIR': './plots',
    'OUTPUT_DIR': './',
    'PRECIP_DATA_FILE': '/public/home/zhaox_hx/data/aidata/zhangym_tmp/DATA/ML_dataset/046_DataSet/pp.csv',

    'SEL_STATION_ID': 2,
    'RAIN_LEVELS': {
                    1: 0.1,  # mm in 24 hours
                    2: 10,
                    3: 25,
                    4: 50,
                    5: 100,
                    6: 250,
                },
    'OPTIMAL_PERCENTILES': {
                    0.1: 6,  # % among ensemble members
                    10: 54,
                    25: 86,
                    50: 92,
                    100: 100,
                    250: 100
                },
    'N_PERCENTILES': 95,
    'N_ENSEMBLES': 20,
    'MIN_ENSEMBLE': 5,
    'LEAD_TIME': 6,  # hours
    'FCST_COLUMN': 'p6',
}
