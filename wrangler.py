import pickle
import sys

import hdbscan
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon
from sklearn.preprocessing import OneHotEncoder

from clustering_helper_funcs import get_hyperparameters, detect_outliers
from shape_helper_funcs import get_polygons


# Wrangler class, used for wrangling data into formats and calculating features.
# This class is very messy at the moment.
# In short, it has methods to calculate different supporting DataFrames, calculate and extract data from trajectories,
# load and dump files etc. All static methods must be run before init_attributes and get_nndf.
# The main functionality is init_attributes -> pdf and get_nndf -> nndf.
# pdf is a DataFrame where each trajectory is a row, and is stored as lists in the DataFrame.
# nndf is a DataFrame where each frame is a row, used for training models.
class Wrangler:
    def __init__(self, df, l_xy, l_df, light_dict: dict = None, direction_dict: dict = None):
        """
        init Wrangler with dfs for wrangling and dictionaries for mapping.
        if no dicts are given, use default

        :param df: trajectory df
        :param l_xy: light positions
        :param l_df: light timings df
        :param light_dict: cluster: light dict
        :param direction_dict:
        """
        self.df = df
        self.l_xy = l_xy
        self.l_df = l_df

        self.pdf = None
        self.nndf = None

        # set mapping of cluster : light, use default if None
        # NOTE THIS MAPPING CAN CHANGE DEPENDING ON MANIPULATIONS OF THE DATA
        self.light_dict = light_dict
        if light_dict is None:
            # self.light_dict = {0: 7, 1: 6, 2: 5, 3: 9, 4: 8, 5: 6, 6: 10, 7: 6, 8: 4, 9: 4, 10: 11, 11: 5}
            self.light_dict = {0: 7, 1: 5, 2: 6, 3: 9, 4: 8, 5: 6, 6: 4, 7: 10, 8: 6, 9: 4, 10: 11, 11: 5}

        # set mapping of cluster : direction, use default if None
        # NOTE THIS MAPPING CAN CHANGE DEPENDING ON MANIPULATIONS OF THE DATA
        self.direction_dict = direction_dict
        if direction_dict is None:
            # self.direction_dict = {
            #     0: 'right', 1: 'straight', 2: 'left', 3: 'left', 4: 'straight', 5: 'left',
            #     6: 'right', 7: 'right', 8: 'straight', 9: 'left', 10: 'right', 11: 'straight'
            # }
            self.direction_dict = {
                                      0: 'right', 1: 'left', 2: 'straight', 3: 'left', 4: 'straight', 5: 'left',
                                      6: 'left', 7: 'right', 8: 'right', 9: 'straight', 10: 'right', 11: 'straight'
            }

    @staticmethod
    def _euc(x, y):
        """
        calculates euclidean distances between neighboring points from xy-coordinates

        :param x:
        :param y:
        :return: distances
        """
        v = np.vstack((x, y)).T
        distances = np.linalg.norm(v[:-1] - v[1:], axis=1)
        return distances

    @staticmethod
    def _get_mid(n):
        """
        calculate mid of multiple points from a df row

        :param n: pandas series w. xy
        :return: midpoint
        """
        return np.array([sum(n['x']) / len(n['x']), sum(n['y']) / len(n['y'])])

    @staticmethod
    def _d2l(x, y, mid):
        """
        calculate distance from xy to a mid point,
        used for calculating distance of current point to traffic light

        :param x:
        :param y:
        :param mid:
        :return: distances to midpoint
        """
        v = np.vstack((x, y)).T
        return np.linalg.norm(v - mid, axis=1)

    @staticmethod
    def get_all_df(df, dump: bool = True, path: str = None):
        """
        get DataFrame with positions of all objects to each frame. must have to init_attributes

        :param df: DataFrame to extract data from
        :param path: path to save location
        :param dump: bool - save file
        :return: all_df
        """
        d = {
            'x': [],
            'y': [],
            'frame': [],
            'class': [],
            'id': []
        }

        for _, row in df.iterrows():
            for i in range(len(row['xs'])):
                d['x'].append(row['xs'][i])
                d['y'].append(row['ys'][i])
                d['frame'].append(row['frames'][i])
                d['class'].append(row['class'])
                d['id'].append(row['id'])

        all_df = pd.DataFrame(d)

        if dump:
            if path is None:
                Wrangler.dump_pickle(all_df, 'alL_df.pkl')
            else:
                Wrangler.dump_pickle(all_df, path)

        return all_df

    @staticmethod
    def cut_ends(df, poly, threshold=50):
        """
        Cut ends of car trajectories, remove wiggly parts

        :param df: DataFrame
        :param poly: Polygon, zone to keep
        :param threshold: int, min number of points in zone
        :return:
        """

        # init dict to use for DataFrame
        cols = ['id', 'class', 'xs', 'ys', 'frames', 'x0', 'y0', 'x1', 'y1']
        d = {c: [] for c in cols}

        # for each car trajectory in trajectories
        for _, row in df.loc[df['class'] == 'Car'].iterrows():
            # track how many points are in the zone
            in_list = []
            for i, (x, y) in enumerate(zip(row['xs'], row['ys'])):
                p = Point([x, y])
                if poly.contains(p):
                    in_list.append(i)

            # if there are more points in the zone than a threshold, add trajectory to DataFrame
            if len(in_list) > threshold:
                d['id'].append(row['id'])
                d['class'].append(row['class'])
                xs = [row['xs'][i] for i in in_list]
                ys = [row['ys'][i] for i in in_list]
                d['xs'].append(xs)
                d['ys'].append(ys)
                d['frames'].append([row['frames'][i] for i in in_list])
                d['x0'].append(xs[0])
                d['x1'].append(xs[-1])
                d['y0'].append(ys[0])
                d['y1'].append(ys[-1])
        return pd.DataFrame(d)

    def init_attributes(self, all_df, step_size: int = 1, num_zones: int = 20, dump: bool = False, path: str = None):
        """
        wrangle data and calculate various attributes from DataFrame,
        must be done before get_nndf,
        can also be loaded with load_pdf

        :param all_df: DataFrame of all xy for all objects to each frame
        :param num_zones: int, number of fields of vision
        :param path: path to save location
        :param dump: bool - save file
        :param step_size: use every step_size frames
        :return: pdf
        """
        d = {
            'x': [],
            'y': [],
            'euc': [],
            'd_t-1': [],
            'd_t-2': [],
            'd_t-3': [],
            'd_light': [],
            'l0': [],
            'l1': [],
            'l2': [],
            'l3': [],
            'dir_0': [],
            'dir_1': [],
            'dir_2': [],
            'frames': [],
            'cluster': [],
            'id': [],
            'light_index': [],
            'class': [],
            'light_color': [],
            'direction': []

        }

        # init one-hot-encoders for categorical variables
        l_enc = OneHotEncoder(handle_unknown='ignore').fit(np.array([0, 1, 2, 3]).reshape(-1, 1))
        d_enc = OneHotEncoder(handle_unknown='ignore').fit(np.array(['left', 'straight', 'right']).reshape(-1, 1))
        classes = np.unique(self.df['class']).reshape(-1, 1)
        c_enc = OneHotEncoder(handle_unknown='ignore').fit(classes)

        # init all classes
        for i, n in enumerate(classes):
            d['c_' + str(i)] = []

        # init zones
        for i in range(num_zones):
            d['d_zone_' + str(i)] = []
            d['zone_' + str(i)] = []

        # for every trajectory, extract and calculate features
        for df_index, row in self.df.iterrows():
            try:
                print(df_index)
                # get x and y, [::step_size] gives every step_size frame
                rowx = np.array(row['xs'][::step_size])
                rowy = np.array(row['ys'][::step_size])

                # get frames
                frames = np.array(row['frames'][::step_size][:-1])

                # calculate midpoint of corresponding traffic light, used for calculating a cars distance to light
                l_mid = self._get_mid(self.l_xy[self.light_dict[row['cluster']]])

                # light color for each trajectory's corresponding traffic light
                l_color = np.array([self.l_df.loc[f][str(self.light_dict[row['cluster']])] for f in frames])

                # one-hot-encode classes, redundant since we only use 'car' class
                encoding = c_enc.transform([[row['class']]]).toarray()
                for i in range(classes.shape[0]):
                    d['c_' + str(i)].append(encoding[0, i])

                # add light_index of trajectory as column
                d['light_index'].append(self.light_dict[row['cluster']])

                # add x and y, not the last coordinate because there is no target for the last step
                d['x'].append(rowx[:-1])
                d['y'].append(rowy[:-1])

                # add id of trajectory
                d['id'] = row['id']

                # add light color and distance to light, light color is one-hot-encoded
                d['light_color'].append(l_color)
                d['d_light'].append(self._d2l(rowx[:-1], rowy[:-1], l_mid))
                encoding = l_enc.transform(l_color.reshape(-1, 1)).toarray()
                for n in range(4):
                    d['l' + str(n)].append(encoding[:, n])

                # calculate and add euclidean distances, target of future training
                eucs = self._euc(rowx, rowy)
                d['euc'].append(eucs)

                # add frames, class, cluster, direction
                d['frames'].append(frames)
                d['class'].append(row['class'])
                d['cluster'].append(row['cluster'])
                direction = self.direction_dict[row['cluster']]
                d['direction'].append(direction)
                encoding = d_enc.transform([[direction]]).toarray()
                for n in range(3):
                    d['dir_' + str(n)].append([encoding[0, n]]*(len(rowx)-1))

                # columns for previous distances, default to 0 if no previous distance exists
                d_t1 = []
                d_t2 = []
                d_t3 = []
                for i in range(len(rowx) - 1):
                    if i >= 1:
                        d_t1.append(eucs[i - 1])
                    else:
                        d_t1.append(0)
                    if i >= 2:
                        d_t2.append(eucs[i - 2])
                    else:
                        d_t2.append(0)
                    if i >= 3:
                        d_t3.append(eucs[i - 3])
                    else:
                        d_t3.append(0)
                d['d_t-1'].append(d_t1)
                d['d_t-2'].append(d_t2)
                d['d_t-3'].append(d_t3)

                # get polygons as vision mechanic for car
                all_polygons = []
                for i in range(len(rowx) - 1):
                    v1 = np.array([rowx[i], rowy[i]])  # current position
                    v_next = np.array([rowx[i+1], rowy[i+1]])  # next position

                    try:
                        count = 1
                        # if current position and next position are equal, use the next position + t as next position
                        while np.all(v1 == v_next):
                            v_next = np.array([rowx[i+1+count], rowy[i+1+count]])
                            count += 1
                        # get num_zones polygons 360 degrees around the car
                        polygons = get_polygons(v1, v_next, num_zones)
                        all_polygons.append(polygons)
                    except IndexError:
                        all_polygons.append(polygons)

                all_polygons = np.array(all_polygons)

                # add polygons to DataFrane
                for i in range(num_zones):
                    d['zone_' + str(i)].append(all_polygons[:, i])

                # with the polygons known, calculate if an object exists inside a zone to each frame using all_df
                # generated using get_all_df method
                d_zone = [[] for _ in range(num_zones)]
                for i, frame in enumerate(frames):
                    p = Point([rowx[i], rowy[i]])  # current point
                    all_df_f = all_df.loc[all_df['frame'] == frame]  # get all objects in current frame
                    # for each zone, check if an object exists within it and return distance
                    for z in range(num_zones):
                        zone = all_polygons[i, z]  # current zone
                        d_f = [[] for _ in range(num_zones)]
                        for _, xy in all_df_f.iterrows():
                            pz = Point([xy['x'], xy['y']])  # objects position
                            if zone.contains(pz):  # if zone contains object
                                d_f[z].append(p.distance(pz))  # get distance to object
                        try:  # append closest object to df
                            d_zone[z].append(min(d_f[z]))
                        except ValueError:  # if no objects are in zone, default to 1000
                            d_zone[z].append(1000)

                # add data to DataFrame
                for z in range(num_zones):
                    d['d_zone_' + str(z)].append(d_zone[z])

            except RuntimeError:
                continue

        # get DataFrame
        self.pdf = pd.DataFrame(d)

        # dump file if desired
        if dump:
            if path is None:
                self.dump_pickle(self.pdf, 'pdf.pkl')
            else:
                self.dump_pickle(self.pdf, path)

        return self

    def load_pdf(self, path: str = 'data/pdf.pkl'):
        """
        load pdf instead of running init_attributes

        :param path: path to pdf
        :return:
        """
        self.pdf = self.load_pickle(path)
        return self

    def get_nndf(self, num_zones: int = 20, dump: bool = False, path: str = None):
        """
        get data formatted for training models,
        must run init_attributes or load_pdf before this method

        :param num_zones: int, number of zones in pdf
        :param dump: save file
        :param path: path to file
        :return: nndf
        """
        if self.pdf is None:
            raise RuntimeError('Must init pdf with init_attributes or load_pdf first')

        # init dictionary for DataFrame
        d = {
            'index': [],
            'x': [],
            'y': [],
            'd_t-1': [],
            'd_t-2': [],
            'd_t-3': [],
            'd_light': [],
            'l0': [],
            'l1': [],
            'l2': [],
            'l3': [],
            'dir_0': [],
            'dir_1': [],
            'dir_2': [],
            'frame': [],
            'cluster': [],
            'class': [],
            'target': []
        }

        # init class columns
        classes = np.unique(self.df['class']).reshape(-1, 1)
        for i, n in enumerate(classes):
            d['c_' + str(i)] = []

        # init zone columns
        for i in range(num_zones):
            d['d_zone_' + str(i)] = []
            d['zone_' + str(i)] = []

        # for every trajectory, make each frame its' own row
        for index, row in self.pdf.iterrows():
            # start from 3 since we have previous distance information
            # arguably try starting from 0 to see if it produces better results
            for i in range(3, len(row['x'])):
                # for every frame, add data to DataFrame
                d['index'].append(index)
                d['x'].append(row['x'][i])
                d['y'].append(row['y'][i])
                d['d_t-1'].append(row['d_t-1'][i])
                d['d_t-2'].append(row['d_t-2'][i])
                d['d_t-3'].append(row['d_t-3'][i])
                d['d_light'].append(row['d_light'][i])
                d['frame'].append(row['frames'][i])
                d['cluster'].append(row['cluster'])
                d['class'].append(row['class'])
                d['target'].append(row['euc'][i])

                for n in range(4):
                    d['l' + str(n)].append(row['l' + str(n)][i])

                for n in range(3):
                    d['dir_' + str(n)].append(row['dir_' + str(n)][i])

                for n in range(len(classes)):
                    d['c_' + str(n)].append(row['c_' + str(n)])

                for n in range(num_zones):
                    d['d_zone_' + str(n)].append(row['d_zone_' + str(n)][i])
                    d['zone_' + str(n)].append(row['zone_' + str(n)][i])

        self.nndf = pd.DataFrame(d)  # create DataFrame

        # save file if desired
        if dump:
            if path is None:
                self.dump_pickle(self.nndf, 'nndf.pkl')
            else:
                self.dump_pickle(self.nndf, path)

        return self

    @staticmethod
    def load_pickle(path: str):
        """
        loads a binary pickle file from a path

        :param path: path to file
        :return: loaded object
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def dump_pickle(obj, path: str):
        """
        dumps an object to a binary pickle file

        :param obj:
        :param path: path to file
        :return:
        """
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def filter_class(df, classes: [str]):
        """
        mask df to get only desired classes,
        must have column 'class'

        :param df:
        :param classes: List of str
        :return: df.loc[mask], mask
        """
        mask = pd.Series([False for _ in range(df.shape[0])])
        for c in classes:
            mask |= df['class'] == c
        return df.loc[mask], mask

    @staticmethod
    def remove_outlier_points_traj(df):
        """
        removes trajectories with above 95 percentile number of points.
        hopefully removes wrongly tracked or outlier trajectories

        :param df: df
        :return: df
        """
        for c in np.unique(df['cluster']):
            mask = df['cluster'] == c
            lens = np.array([len(row['xs']) for _, row in df.loc[mask].iterrows()])
            cut = np.percentile(lens, 95)
            remove = lens > cut
            idx = df.loc[mask].loc[remove].index
            df = df.drop(idx)
        return df

    @staticmethod
    def add_trajectory_lens(df):
        """
        adds trajectory lengths to DataFrame

        :param df: DataFrame
        :return: DataFrame, with added column
        """
        lengths = []
        for _, row in df.iterrows():
            trajectory = np.vstack(row[['xs', 'ys']].to_numpy()).T
            length = Wrangler.trajectory_len(trajectory)
            lengths.append(length)
        df['traj_lengths'] = lengths
        return df

    @staticmethod
    def remove_long_traj(df):
        """
        for each cluster, remove the longest ones

        :param df: DataFrame
        :return: DataFrame, with removed rows
        """
        for c in np.unique(df['cluster']):
            mask = df['cluster'] == c
            lens = df.loc[mask, 'traj_lengths']
            cut = np.percentile(lens, 90)
            remove = lens > cut
            idx = df.loc[mask].loc[remove].index
            df = df.drop(idx)
        return df

    @staticmethod
    def trajectory_len(trajectory, total_length=0):
        """
        calculates length of trajectory

        :param trajectory: np.array, trajectory with shape (-1,2)
        :param total_length: 0, length of trajectory, init to 0
        :return: float, total length of trajectory
        """
        if trajectory.shape[0] <= 1:
            return total_length
        a = trajectory[0]
        b = trajectory[1]
        length = np.linalg.norm(a - b)
        total_length += length
        return Wrangler.trajectory_len(trajectory[1:], total_length)


# if running wrangler.py, wrangle data into pdf and nndf using methods from Wrangler
if __name__ == '__main__':
    # --------load raw data--------
    # load raw trajectory data
    df = Wrangler.load_pickle('bsc-3m/traj_01_elab.pkl')
    df_frames = Wrangler.load_pickle('bsc-3m/traj_01_elab_new.pkl')
    df = df.join(df_frames['frames'])

    # load traffic lights coordinates and color info
    l_xy = Wrangler.load_pickle('bsc-3m/signal_lines_true.pickle')
    l_df = pd.read_csv('bsc-3m/signals_dense.csv')

    # --------preprocess before creating pdf and nndf--------
    # get all_df
    # all_df = Wrangler.get_all_df(df, dump=True, path='data/all_df.pkl')
    all_df = Wrangler.load_pickle('data/all_df.pkl')

    # select strictly cars, remove later?
    df, _ = Wrangler.filter_class(df, ['Car'])

    # cut ends
    points = [
        Point([650, 100]), Point([850, 200]), Point([1050, 100]), Point([1200, 200]), Point([1100, 450]),
        Point([1025, 525]), Point([550, 600]), Point([400, 600]), Point([100, 550]), Point([100, 475]), Point([60, 350])
    ]
    poly = Polygon(points)
    df = Wrangler.cut_ends(df, poly)

    # cluster and remove outliers
    # HDBSCAN for now, try other in future?
    # should be added as method to Wrangler
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

    # removing trajectories with highest number of points
    fdf = Wrangler.remove_outlier_points_traj(fdf)

    # remove trajectories that are too long for their cluster
    sys.setrecursionlimit(10000)
    fdf = Wrangler.add_trajectory_lens(fdf)
    fdf = Wrangler.remove_long_traj(fdf)

    # plot all clusters in figure and save in figs/
    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(40, 40))
    img = Image.open("intersection2.png")
    img = ImageOps.flip(img)

    for r in range(4):
        for c in range(3):
            cluster = r * 3 + c
            ax[r, c].set_title(f'cluster {cluster}')
            ax[r, c].set_xlim(0, 1280)  # set x-range to resolution
            ax[r, c].set_ylim(0, 720)  # set y-range to resoultion
            ax[r, c].imshow(img, origin='lower')
            mask = fdf['cluster'] == cluster
            for _, row in fdf.loc[mask].iterrows():
                ax[r, c].plot(row['xs'], row['ys'], c='b', alpha=0.05)
            ax[r, c].get_xaxis().set_visible(False)  # hide x-axis
            ax[r, c].get_yaxis().set_visible(False)  # hide y-axis
    plt.savefig('figs/clusters.svg')

    # --------create pdf and nndf--------
    print('start')
    # wrangle data into shape
    # wr = Wrangler(fdf, l_xy, l_df) \
    #     .load_pdf('data/pdf_zones.pkl') \
    #     .get_nndf(dump=True, path='data/nndf.pkl')
    wr = Wrangler(fdf, l_xy, l_df) \
        .init_attributes(all_df, step_size=5, dump=True, path='data/pdf.pkl') \
        .get_nndf(dump=True, path='data/nndf.pkl')
