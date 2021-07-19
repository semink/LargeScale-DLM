from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import expm
from lsdlm import bayesian as bay
from tqdm import tqdm
from itertools import cycle, islice
import pickle, os


class DLM:
    def __init__(self, adj_mx, num_diff_periods=5, tau=None):
        """
        :param scaler: 
        :param adj_mx: 
        :param num_diff_periods: 
        :param H_mode: 
        :param diffusion_mode: 
        """
        self.H = dict()
        self.params = dict()
        self.adj_mx = adj_mx
        self.scaler = StandardScaler()

        # laplacian matrix
        self.L = csgraph.laplacian(self.adj_mx, normed=False).T
        if tau is None: self.tau = self._search_tau(num_diff_periods)
        else: self.tau = tau
        self.M = np.stack([expm(-tau * self.L) for tau in self.tau], axis=2)

    def _search_tau(self, num_diff_periods, ep=1e-5):
        T = range(-10, 10)
        N = self.L.shape[0]
        for i, tau in enumerate(T):
            if 1 / N * np.linalg.norm(expm(-(10 ** tau) * self.L) - expm(-(10 ** T[0]) * self.L)) > ep:
                print(f'tau_short = {10 ** tau}')
                break
        tau_short = T[i - 1]
        for tau in range(-10, 10):
            if 1 / N * np.linalg.norm(expm(-(10 ** tau) * self.L) - expm(-(10 ** T[-1]) * self.L)) < ep:
                print(f'tau_long = {10 ** tau}')
                break
        tau_long = tau
        tau = np.logspace(tau_short, tau_long, num_diff_periods)
        return tau

    def _mulH(self, x):
        hour = x.name.strftime("%H:%M")
        H = self.H[hour]
        return H @ x

    @staticmethod
    def _exclude_nan(Vt, Vtp1):
        vt_valid_idx = ~np.isnan(Vt).any(axis=1)
        vtp1_valid_idx = ~np.isnan(Vtp1).any(axis=1)
        valid_idx = vt_valid_idx & vtp1_valid_idx
        return Vt[valid_idx], Vtp1[valid_idx]

    def fit(self, train_df):
        '''
        train_df: [number of samples X number of sensors]
        '''
        df_scaled = pd.DataFrame(self.scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index)

        def fit_hour(Vt, Vtp1, hour):
            Vt, Vtp1 = self._exclude_nan(Vt, Vtp1)
            X, T = Vt.T, Vtp1.T
            self.H[hour], success, self.params[hour] = bay.get_optimal_H(X, T, self.M)
            if not success: print(f'fail to train H for {hour}.')

        self.hour_vec = pd.to_datetime(list(set(train_df.index.strftime('%H:%M')))).sort_values().strftime(
            '%H:%M').values
        dt = train_df.index[1] - train_df.index[0]

        for i, hour in enumerate(tqdm(self.hour_vec)):
            Vt = df_scaled.at_time(hour)  # ex) 01-01 23:55, 01-02 23:55, ..., 03-31 23:55
            valid_idx = (Vt.index + dt).isin(
                df_scaled.index)  # ex) 01-02 00:00, 01-03 00:00, ..., 04-01 00:00 -> True, True, ..., False
            Vtp1 = df_scaled.loc[(Vt.index + dt)[valid_idx]].values  # ex) 01-02 00:00, 01-03 00:00, ..., 03-31 00:00
            Vt = df_scaled.loc[Vt.index[valid_idx]].values  # ex) 01-01 23:55, 01-02 23:55, ..., 03-30 23:55
            fit_hour(Vt, Vtp1, hour)

    def predict(self, df, step_ahead=1):
        dfs = []
        df = pd.DataFrame(self.scaler.transform(df), index=df.index, columns=df.columns)
        for i, hour in tqdm(enumerate(self.hour_vec)):
            df_hour = df.at_time(hour)
            Hs = [self.H[circular_hour] for circular_hour in islice(cycle(self.hour_vec), i, i + step_ahead)]
            H = np.linalg.multi_dot(Hs[::-1])       # should be reverse order!!
            dfs.append((H @ df_hour.T).T)
        df_pred = pd.concat(dfs).sort_index().shift(step_ahead).dropna()
        return pd.DataFrame(self.scaler.inverse_transform(df_pred), index=df_pred.index, columns=df.columns)

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
