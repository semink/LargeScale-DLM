from lsdlm import utils, lsdlm
import pickle
import numpy as np


def train():
    print('loading dataset...', end=' ')
    df_raw, df_meta = utils.load_dataset()
    df_raw.columns = df_raw.columns.astype(df_meta.index.dtype)
    df_raw = df_raw[df_meta.index]
    df_clean = utils.preprocess(df_raw, replace={'from': 0, 'to': np.NaN})
    print('done.')

    print('splitting dataset to training and test set...', end=' ')
    df_train, df_test = utils.split_dataset(df_clean)
    print('done.')

    # if you want to train a model, uncomment following part
    adj = utils.load_graph(df_meta, path='data/adj_mx_bay.pkl')
    model = lsdlm.DLM(adj_mx=np.maximum(adj, adj.T), num_diff_periods=5)        # undirected graph
    print('model created... start to train...')
    model.fit(df_train)
    model.save_model('data', 'pretrained.model')
    model = pickle.load(open('data/pretrained.model', 'rb'))

    for step_ahead in [3, 6, 12]:
        print(f'prediction for h={step_ahead}...')
        df_pred = model.predict(df_test, step_ahead=step_ahead)
        print(f'RMSE: {np.sqrt(((df_test - df_pred) ** 2).mean().mean())}')


if __name__ == '__main__':
    train()
