import numpy as np
import pandas as pd


def get_hyperparameters(classes: str, algorithm: str, cluster: int = -1) -> tuple:
    if algorithm == 'hdbscan':
        # Tested HDBSCAN hyper-parameters
        # min_cluster_size, min_samples, cluster_selection_epsilon
        if classes == 'Bicycle':
            if cluster > -1:
                return 10, 10, 10
            # 20, 20, 10 - 29 clusters
            # 50, 50, 10 - 21 clusters
            # 100, 100, 10 - 16 clusters
            return 20, 20, 10
        elif classes == 'Car':
            # 50, 50, 10 - X clusters
            return 50, 50, 5
        elif classes == 'Pedestrian':
            # 5, 5, 10 - 23 clusters
            # 10, 10, 10 - 13 clusters
            return 5, 5, 10
        elif classes == 'Motorcycle':
            # 10, 10, 10 - X clusters
            return 10, 10, 10
        elif classes == 'Van':
            # 10, 10, 10 - X clusters
            return 10, 10, 10
        elif classes == 'Light Truck':
            # 5, 5, 10 - X clusters
            return 5, 5, 10
        elif classes == 'Heavy Vehicle':
            # 5, 5, 10 - X clusters
            return 5, 5, 10
        elif classes == 'Bus':
            # 2, 2, 10 - X clusters
            return 2, 2, 10
    else:
        # !!!!! Implement other clustering algorithms
        pass
    
    return 100, 100, 10


def detect_outliers(clusterer, df):
    # Locating all outliers.
    threshold = pd.Series(clusterer.outlier_scores_).quantile(0.97)  # threshold for cutoff.
    outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
    df["outliers"] = clusterer.outlier_scores_
    df_o = df[df["outliers"] > threshold]
    return df_o, df
