import pickle
import pandas as pd
import numpy as np
import configparser

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import seaborn as sns

import hdbscan

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from wrangler import Wrangler
from clustering_helper_funcs import *


def main():
    config = configparser.ConfigParser()
    config.read('settings.ini')
    config = config['DEFAULT']

    wrangle_bool = config.getboolean('wrangle')
    if wrangle_bool:
        # load trajectory dataframes from pickle and merge to get frame data
        print('loading data...')
        df = Wrangler.load_pickle('bsc-3m/traj_01_elab.pkl')
        df_frames = Wrangler.load_pickle('bsc-3m/traj_01_elab_new.pkl')
        df = df.join(df_frames['frames'])

        all_df = Wrangler.get_all_df(df, dump=True, path='data/all_df.pkl')

        # load traffic lights coordinates and color info
        l_xy = Wrangler.load_pickle('bsc-3m/signal_lines_true.pickle')
        l_df = pd.read_csv('bsc-3m/signals_dense.csv')

        # select strictly cars, remove later?
        df, _ = Wrangler.filter_class(df, ['Car'])

        # cluster and remove outliers
        # HDBSCAN for now, try other in future?
        print('clustering...')
        min_cluster_size, min_samples, cluster_selection_epsilon = get_hyperparameters('Car', '')
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon
        )
        x = df[['x0', 'y0', 'x1', 'y1']].to_numpy()  # prepare data for clustering
        xc = np.array(clusterer.fit_predict(x))
        df['cluster'] = xc
        fdf = detect_outliers(clusterer, df)
        fdf = fdf.loc[fdf['cluster'] != -1]

        # wrangle data into shape
        print('wrangling data...')
        wr = Wrangler(fdf, l_xy, l_df)\
            .init_attributes(all_df, step_size=config.getint('step_size'), dump=config.getboolean('dump_data'),
                             path=config['data_path']+'pdf.pkl')\
            .get_nndf(dump=config.getboolean('dump_data'), path=config['data_path']+'nndf.pkl')
        nndf = wr.nndf

    else:
        pdf = Wrangler.load_pickle(config['data_path'] + 'pdf.pkl')
        nndf = Wrangler.load_pickle(config['data_path'] + 'nndf.pkl')

    plot_bool = config.getboolean('plot')
    if plot_bool:
        pass  # plot beautiful stuff

    load_model = config.getboolean('load_model')
    if not load_model:
        # rewrite split to use whole trajectories
        print('preparing data...')
        cols = ['x', 'y', 'd_t-1', 'd_t-2', 'd_t-3', 'd_light', 'l0', 'l1',
                'l2', 'l3', 'dir_0', 'dir_1', 'dir_2', 'target']
        df_train, df_val = train_test_split(nndf[cols], test_size=0.2, random_state=1)
        df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=1)

        x_train, y_train = df_train[cols[:-1]].to_numpy(), df_train[cols[-1:]].to_numpy().reshape(-1)
        x_val, y_val = df_val[cols[:-1]].to_numpy(), df_val[cols[-1:]].to_numpy().reshape(-1)
        x_test, y_test = df_test[cols[:-1]].to_numpy(), df_test[cols[-1:]].to_numpy().reshape(-1)

        print('training model...')
        clf = MLPRegressor(verbose=True).fit(x_train, y_train)
        score = clf.score(x_val, y_val)
        print('validation score:\t', score)

        dump_model = config.getboolean('dump_model')
        if dump_model:
            model_str = 'model.pkl'
            Wrangler.dump_pickle(clf, config['model_path']+model_str)

    else:
        model_str = 'model.pkl'
        clf = Wrangler.load_pickle(config['model_path']+model_str)

    simulate = config.getboolean('simulate')
    if simulate:
        pass


if __name__ == '__main__':
    main()
