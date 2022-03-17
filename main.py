import pickle
import pandas as pd
import numpy as np
import configparser

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import seaborn as sns

import hdbscan

from wrangler import Wrangler
from clustering_helper_funcs import *


def main():
    config = configparser.ConfigParser()
    config.read('settings.ini')
    config = config['DEFAULT']

    if config['wrangle']:
        # load trajectory dataframes from pickle and merge to get frame data
        print('loading data...')
        df = Wrangler.load_pickle('bsc-3m/traj_01_elab.pkl')
        df_frames = Wrangler.load_pickle('bsc-3m/traj_01_elab_new.pkl')
        df = df.join(df_frames['frames'])

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
            .init_attributes(dump=config['dump'], path=config['path']+'pdf.pkl')\
            .get_nndf(dump=config['dump'], path=config['path']+'pdf.pkl')
        nndf = wr.nndf

    else:
        pdf = Wrangler.load_pickle(config['path'] + 'pdf.pkl')
        nndf = Wrangler.load_pickle(config['path'] + 'nndf.pkl')

    if config['plot']:
        pass  # plot beautiful stuff


if __name__ == '__main__':
    main()
