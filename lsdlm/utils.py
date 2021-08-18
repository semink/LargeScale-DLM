import os, sys, logging
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


root = get_project_root()


def load_pretrained_model():
    url = 'https://zenodo.org/record/4264445/files/pretrained.pkl'
    return pd.read_pickle(url)


def load_dataset(name='PEMS-BAY', replace_nan=np.NaN, freq='5T'):
    if not os.path.isfile(os.path.join(root, 'data/PEMS-BAY.csv')):
        url_data = 'https://zenodo.org/record/4264005/files/PEMS-BAY.csv'
        pd.read_csv(url_data, index_col=0).to_csv(os.path.join(root, 'data/PEMS-BAY.csv'))
    if not os.path.isfile('data/PEMS-BAY-META.csv'):
        url_meta = 'https://zenodo.org/record/4264005/files/PEMS-BAY-META.csv'
        pd.read_csv(url_meta, index_col=0).to_csv(os.path.join(root, 'data/PEMS-BAY-META.csv'))

    df_raw = pd.read_csv(os.path.join(root,'data/PEMS-BAY.csv'), index_col=0)
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw = df_raw.resample(freq).asfreq().fillna(replace_nan)

    df_meta = pd.read_csv(os.path.join(root,'data/PEMS-BAY-META.csv'), index_col=0)

    return df_raw, df_meta


def load_graph(df_meta, path=os.path.join(root,'data/adj_mx_bay.pkl')):
    with open(path, 'rb') as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')

    order = [sensor_id_to_ind[str(sensor)] for sensor in df_meta.index]
    adj = np.zeros((df_meta.shape[0], df_meta.shape[0]))
    for i, o_r in enumerate(order):
        for j, o_c in enumerate(order):
            adj[i, j] = adj_mx[o_r, o_c]

    return adj


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)
        # Add file handler and stdout handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
        file_handler.setFormatter(formatter)
        # Add console handler.
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def mask_by_time(df, time_between, offset=0):
    mask = []
    for time in time_between:
        [mask.append(i + offset) for i in df.index.indexer_between_time(start_time=time[0], end_time=time[1])]
    mask = [x for x in mask if x < df.shape[0]]
    return df.iloc[mask]


def preprocess(df, threshold={'max': 90, 'min': 0},
               replace={'from': np.NaN, 'to': np.NaN}):
    # threshold cut
    def cut(x):
        x[x > threshold['max']] = threshold['max']
        x[x < threshold['min']] = threshold['min']
        return x

    tqdm.pandas(desc='Threshold cutting')
    df = df.progress_apply(cut)

    # replace
    df = df.replace(replace['from'], replace['to'])

    # fill missing index
    t_samples = pd.to_datetime(list(set(df.index.strftime('%H:%M')))).sort_values()
    dt = t_samples.to_series().diff().min()
    df = df.resample(dt).asfreq()
    return df


def split_dataset(df, rule={'train': 8, 'test': 2}, cut_by_day=True):
    if cut_by_day:
        days = list(dict.fromkeys(df.index.strftime('%Y-%m-%d')))
        n_of_days = len(days)
        train_cut = int(n_of_days * (rule['train'] / (rule['train'] + rule['test'])))
        train_df, test_df = df[:days[train_cut]], df[days[train_cut + 1]:]
    else:
        n = df.shape[1]
        train_cut = int(n * (rule['train'] / (rule['train'] + rule['test'])))
        train_df, test_df = df.iloc[:train_cut], df.iloc[train_cut + 1:]
    return train_df, test_df
