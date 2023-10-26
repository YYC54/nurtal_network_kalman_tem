from sklearn.model_selection import train_test_split
from lib.Thunder.code.testCorr import Thunder, processer, create_deep_model, lr_schedule, CustomDataset
import pandas as pd


# Logger = Logger(path="./log/tem-result.log", clevel=logging.ERROR, Flevel=logging.DEBUG)


class Thunder_MODEL(object):
    """浅层风的订正方法类"""

    def __init__(self, factime, dataframe, element, methods):
        self.factime = factime
        self.dataframe = dataframe
        self.eleent = element
        self.methods = methods
        self.Thuder_base_fea = ['dtime', 'msl', 'lsp', 'cape', 'sp', 'u10', 'v10', 'tcwv',
                                't2m', 'capes', '500t', '850t', '50gh', '500gh', '850gh', '500u',
                                '500v', '500r', 'KI', 'SI', 'TT', 'SWEAT']
        self.Th_E = Thunder(stime=self.factime[:-2], order=self.factime[-2:], element=self.eleent,
                            methods=self.methods, base_fea=self.Thuder_base_fea,
                            target_fea='td')

    def _feature(self):
        fe = self.Th_E.feature(a=self.dataframe)
        self.dataframe = fe
        return fe

    def _train(self):
        p_fea = processer(base_fea=self.Thuder_base_fea, target_fea='td')

        fea = p_fea.base_fea + [p_fea.target_fea]
        Th_ds = self.dataframe[fea]

        # 分离特征列和目标列
        X = Th_ds[p_fea.base_fea]
        y = Th_ds[p_fea.target_fea]

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

        self.Th_E.train(X_train, X_valid, y_train, y_valid)

    def _pred(self):
        if 'KI' not in self.dataframe.columns:
            self.dataframe = self._feature()
        self.Th_E.dataframe = self.dataframe
        self.Th_E.pred()
        df_corrres = self.Th_E.dataframe
        df_corrres['stime'] = pd.to_datetime(df_corrres['stime'])
        df_corrres['time'] = pd.to_datetime(df_corrres['time'])
        df_corrres['time_order'] = df_corrres['stime'].dt.hour
        df_corrres['北京时间'] = df_corrres['time'].apply(lambda x: x + pd.Timedelta(8, unit='H'))
        df_corrres = df_corrres[
            ['time', 'time_order', 'dtime', 'id', 'lon', 'lat', 'stime', 'storm', 'random_forest', 'xgboost',
             'neural_network', 'ensemble', '北京时间']]
        df_corrres.rename(columns={'storm': 'ib', 'random_forest': 'rf', 'xgboost': 'xg', 'neural_network': 'nm',
                                   'ensemble': 'co'}, inplace=True)
        return df_corrres


if __name__ == '__main__':
    dataframe = pd.read_csv("/mnt/PRESKY/user/wanghaifeng/Projects/046/code/C1D010100.csv")
    thunder_e = Thunder_MODEL(
        factime="2021010200",
        dataframe=dataframe,
        element="result-Thunder",
        methods=("random_forest", "xgboost", "neural_network", "peiliao", "ensemble"),
    )
    # max_winds_M._train('neural_network')
    thunder_e._feature()
    df_corrres = thunder_e._pred()
    print(df_corrres)
