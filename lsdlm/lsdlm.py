from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import expm
from lsdlm import bayesian as bay
from tqdm import tqdm
from itertools import cycle, islice



class DLM:
    def __init__(self, adj_mx, num_diff_periods=5):
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
        L = csgraph.laplacian(self.adj_mx, normed=False).T
        self.tau = self._search_tau(num_diff_periods)
        self.M = np.stack([expm(-tau * L) for tau in self.tau], axis=2)

    def _search_tau(self, num_diff_periods, ep=1e-5):
        L = csgraph.laplacian(self.adj_mx, normed=False).T
        T = range(-10, 10)
        for i, tau in enumerate(T):
            if 1 / (L.shape[0]) * np.linalg.norm(expm(-(10 ** tau) * L) - expm(-(10 ** T[0]) * L)) > ep:
                print(f'tau_short = {10 ** tau}')
                break
        tau_short = T[i - 1]
        for tau in range(-10, 10):
            if 1 / (L.shape[0]) * np.linalg.norm(expm(-(10 ** tau) * L) - expm(-(10 ** T[-1]) * L)) < ep:
                print(f'tau_long = {10 ** tau}')
                break
        tau_long = tau
        tau = np.logspace(tau_short, tau_long, num_diff_periods)
        return tau

    def _mulH(self, x):
        hour = x.name.strftime("%H:%M")
        H = self.H[hour]
        return H @ x

    def fit(self, train_df):
        '''
        train_df: [number of samples X number of sensors]
        '''
        N = train_df.shape[1]
        df_scaled = pd.DataFrame(self.scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index)

        def fit_hour(Vt, Vtp1, hour):
            i = np.where(self.hour_vec == hour)

            X, T = Vt.T, Vtp1.T
            H, success, self.params[hour] = bay.get_optimal_H(X, T, self.M)
            alpha, gamma = self.params[hour]['alpha'], self.params[hour]['gamma']
            s, _ = np.linalg.eigh(X @ X.T)
            p_data, p_prior = np.linalg.norm(alpha * s / (alpha * s + gamma)), np.linalg.norm(
                gamma / (alpha * s + gamma))
            self.params[hour]['p_data'] = p_data
            self.params[hour]['p_prior'] = p_prior
            self.H[hour] = H
            if not success: print(f'fail to train H for {hour}.')

        self.hour_vec = pd.to_datetime(list(set(train_df.index.strftime('%H:%M')))).sort_values().strftime(
            '%H:%M').values
        dt = train_df.index[1] - train_df.index[0]

        for i, hour in enumerate(tqdm(self.hour_vec)):
            Vt = df_scaled.at_time(hour)
            valid_idx = (Vt.index + dt).isin(df_scaled.index)
            Vtp1 = df_scaled.loc[(Vt.index + dt)[valid_idx]].values
            Vt = df_scaled.loc[Vt.index[valid_idx]].values
            fit_hour(Vt, Vtp1, hour)

    def predict(self, df, step_ahead=1):
        dfs = []
        df = pd.DataFrame(self.scaler.transform(df), index=df.index, columns=df.columns)
        for i, hour in tqdm(enumerate(self.hour_vec)):
            df_hour = df.at_time(hour)
            H = np.linalg.multi_dot([self.H[circular_hour] for circular_hour in islice(cycle(self.hour_vec), i, i+step_ahead)])
            dfs.append((H@df_hour.T).T)
        df_pred = pd.concat(dfs).sort_index()
        return pd.DataFrame(self.scaler.inverse_transform(df_pred), index=df.index, columns=df.columns)

    def evaluate(self, df, max_step=3):
        pred_df = pd.DataFrame(self.scaler.transform(df), index=df.index, columns=df.columns)
        err = dict()
        for step in tqdm(range(max_step)):
            pred_df = pred_df.apply(lambda x: self._mulH(x), axis=1, result_type='broadcast').shift().dropna()
            # error metric
            prd = pd.DataFrame(self.scaler.inverse_transform(pred_df), index=pred_df.index, columns=pred_df.columns)
            err[step + 1] = [(prd - df).abs().mean().mean(),
                             ((prd - df).abs() / df).replace([np.inf, -np.inf], np.nan).mean().mean() * 100,
                             ((prd - df) ** 2).mean().mean() ** .5]
        return pd.DataFrame.from_dict(err, orient='index', columns=['mae [mph]', 'mape [%]', 'rmse [mph]'])

    def save_model(self, path, fn):
        import pickle, os
        fp = os.path.join(path, fn)
        with open(fp, 'wb') as f:
            pickle.dump(self, f)
