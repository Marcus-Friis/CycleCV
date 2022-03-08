import hdbscan
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import seaborn as sns
from clustering_hyperparams import get_hyperparameters


def main():
    with open('bsc-3m/traj_01_elab.pkl', 'rb') as f:
        df = pickle.load(f)
    df = df.loc[df['class'] == 'Car']
    x = df[['x0', 'y0', 'x1', 'y1']].to_numpy()

    min_cluster_size, min_samples, cluster_selection_epsilon = get_hyperparameters('Car', '')
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon
    )
    xc = np.array(clusterer.fit_predict(x))
    df['cluster'] = xc

    with open('bsc-3m/traj_clustered.pkl', 'wb') as f:
        pickle.dump(df, f)
    
    """
    col = sns.color_palette()
    print('plotting all clusters')
    fig, ax = plt.subplots(figsize=(16, 16))
    im = Image.open("intersection2.png")
    im = ImageOps.flip(im)
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 720)
    ax.imshow(im, origin='lower')
    for _, row in df.iterrows():
        try:
            ax.plot(row['xs'], row['ys'], color=col[row['cluster']], alpha=0.3)
        except IndexError:
            continue
    plt.savefig('figs/all_clusters.svg', dpi=150)

    for cluster in np.unique(df['cluster']):
        fig, ax = plt.subplots(figsize=(16, 16))
        im = Image.open("intersection2.png")
        im = ImageOps.flip(im)
        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 720)
        ax.imshow(im, origin='lower')
        print('plotting cluster', cluster)
        df_c = df.loc[df['cluster'] == cluster]
        for _, row in df_c.iterrows():
            ax.plot(row['xs'], row['ys'], color=col[0], alpha=0.3)
        plt.savefig('figs/cluster_' + str(cluster) + '.svg', dpi=150)
    """


if __name__ == '__main__':
    main()
