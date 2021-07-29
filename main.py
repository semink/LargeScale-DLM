from lsdlm import utils, lsdlm
import pickle
import numpy as np
import time
import argparse


def train(training_dataset, weight_matrix, save_to='data/pretrained_PEMS-BAY.model'):
    model = lsdlm.DLM(adj_mx=np.maximum(weight_matrix, weight_matrix.T), num_diff_periods=5)  # undirected graph
    print('model created... start to train...')
    model.fit(training_dataset)
    model.save_model(save_to)
    print('training finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='PEMS-BAY', type=str,
                        choices=['PEMS-BAY', 'METR-LA'])
    parser.add_argument('--horizon', default=3, type=int)
    parser.add_argument('--train', default=True, type=bool)
    args = parser.parse_args()

    df_raw, adj = utils.load_dataset(name=args.dataset, replace_nan=0.0)
    df_train, df_test = utils.split_dataset(df_raw)
    df_test = utils.preprocess(df_test, replace={'from': 0.0, 'to': np.NaN})

    train_model_path = f'data/pretrained_{args.dataset}.model'
    if args.train:
        train(save_to=train_model_path, training_dataset=df_train, weight_matrix=adj)
    # model is saved as it will take around 16 min.

    model = pickle.load(open(train_model_path, 'rb'))
    before = time.time()
    df_pred = model.predict(df_test, step_ahead=args.horizon)
    print(f'RMSE: {np.sqrt(((df_test - df_pred) ** 2).mean().mean()):.2f}\n')
    print(f'Total computation for prediction: {time.time() - before:.2f} sec')
