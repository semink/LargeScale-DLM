from lsdlm import utils, lsdlm
import pickle


def train():
    print('loading dataset...', end=' ')
    df_raw, df_meta = utils.load_dataset()
    print('done.')

    print('splitting dataset to training and test set...', end=' ')
    df_train, df_test = utils.split_dataset(df_raw)
    print('done.')

    # if you want to train a model, uncomment following part
    # adj = utils.load_graph(df_meta, path='data/adj_mx_bay.pkl')
    # model = lsdlm.DLM(adj_mx=adj)
    # print('model created... start to train...')
    # model.fit(df_train)
    # model.save_model('data', 'pretrained.model')

    model = pickle.load(open('data/pretrained.model', 'rb'))

    step_ahead = 12
    print(f'prediction for h={step_ahead}...')
    model.predict(df_test, step_ahead=step_ahead)


if __name__ == '__main__':
    train()
