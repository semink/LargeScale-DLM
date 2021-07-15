from lsdlm import utils, lsdlm


def train():
    print('loading dataset...', end=' ')
    df_raw, df_meta = utils.load_dataset()
    print('done.')

    print('splitting dataset to training and test set...', end=' ')
    df_train, df_test = utils.split_dataset(df_raw)
    print('done.')

    adj = utils.load_graph(df_meta, path='data/adj_mx_bay.pkl')

    model = lsdlm.DLM(adj_mx=adj)
    print('model created... start to train...')
    model.fit(df_train)


if __name__ == '__main__':
    train()
