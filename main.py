from lsdlm import utils, lsdlm
import pickle
import numpy as np
import time


def train(df_train, df_meta, save_to='data/pretrained.model'):
    # if you want to train a model, uncomment following part
    adj = utils.load_graph(df_meta, path='data/adj_mx_bay.pkl')
    model = lsdlm.DLM(adj_mx=np.maximum(adj, adj.T), num_diff_periods=5)  # undirected graph
    print('model created... start to train...')
    model.fit(df_train)
    model.save_model(save_to)
    print('training finished!')


if __name__ == '__main__':
    df_raw, df_meta = utils.load_dataset()
    df_clean = utils.preprocess(df_raw, replace={'from': 0, 'to': np.NaN})
    df_train, df_test = utils.split_dataset(df_clean)

    train_model_path = 'data/pretrained.model'
    train(save_to=train_model_path, df_train=df_train, df_meta=df_meta)  # Please comment this line once a pretrained
    # model is saved as it will take around 16 min.

    model = pickle.load(open(train_model_path, 'rb'))
    before = time.time()
    step_ahead = 12
    df_pred = model.predict(df_test, step_ahead=step_ahead)
    print(f'RMSE: {np.sqrt(((df_test - df_pred) ** 2).mean().mean())}\n')
    print(f'Total computation for prediction: {time.time() - before} sec')
