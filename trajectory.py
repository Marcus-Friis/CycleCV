import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import itertools
from shape_helper_funcs import *
from wrangler import Wrangler


class Trajectory:
    full_sim_data_cols = ['index', 'x', 'y', 'd_t-1', 'd_t-2', 'd_t-3', 'd_light', 'l0', 'l1',
                          'l2', 'l3', 'dir_0', 'dir_1', 'dir_2', 'frame', 'class'] \
                         + ['d_z_'+str(i) for i in range(20)] + ['z_'+str(i) for i in range(20)]
    full_sim_data = pd.DataFrame(columns=full_sim_data_cols)
    l_enc = OneHotEncoder(handle_unknown='ignore').fit(np.array([[0, 1, 2, 3]]).reshape(-1, 1))
    index_gen = itertools.count()

    def __init__(self, data, l_df, l_xy):
        """
        class for simulating trajectories

        :param data: pandas Series from pdf DataFrame
        :param l_df: pandas DataFrame with signals info
        :param l_xy: list of dicts with xy for signals
        :param clf: trained classifier implementing sklearn interface
        """
        # data and clf for wrangling and predicting
        self.data = data
        self.l_df = l_df
        self.l_xy = l_xy

        # trajectory variables for full trajectory and remaining when simulating
        self.traj_full = np.array([data['x'], data['y']]).T
        self.traj_rest = self.traj_full[1:]
        self.xy_hist = []

        # variables for getting relevant light info
        self.light_index = int(self.data['light_index'])
        n = self.l_xy[self.light_index]
        self.light_mid = np.array([sum(n['x']) / len(n['x']), sum(n['y']) / len(n['y'])])

        # the simulated data and history of distances
        self.sim_data = None
        self.distances = []
        self.c = self.data['class']
        self.index = next(Trajectory.index_gen)

    def init_sim(self, frame: int, i: int = 0):
        """
        initialize simulation and format data

        :param frame: int, for getting signal color
        :param i: int, which index to start from
        :return: DataFrame, ready to simulate
        """
        d = {'index': self.index, 'x': self.data['x'][i], 'y': self.data['y'][i]}  # init index and xy coordinate
        self.xy_hist.append([self.data['x'][i], self.data['y'][i]])

        # init previous distances and save in hist of distances
        for n in ['1', '2', '3']:
            string = 'd_t-' + n
            self.distances.append(self.data[string][i])
            d[string] = self.distances[-1]
        self.distances = self.distances[::-1]

        d['d_light'] = self.data['d_light'][i]  # get dist to light

        # one-hot-encode light color to current frame and add to data
        l_color = self.l_df.loc[frame][str(self.light_index)]
        encoding = self.l_enc.transform([[l_color]]).toarray()
        for n in range(4):
            d['l' + str(n)] = [encoding[0, n]]

        # get direction from data
        for n in ['dir_0', 'dir_1', 'dir_2']:
            d[n] = [self.data[n][i]]

        # add current frame to dataframe
        d['frame'] = [frame]
        d['class'] = [self.c]

        # init zone cols
        for i in range(20):
            d['d_z_'+str(i)] = [np.nan]
        for i in range(20):
            d['z_'+str(i)] = [np.nan]

        # save data, add to DataFrame of all data
        self.sim_data = pd.DataFrame(d)
        Trajectory.full_sim_data = pd.concat((Trajectory.full_sim_data, self.sim_data))
        return self.sim_data

    def get_d_zones(self, frame):
        # zones
        mask = self.sim_data['frame'] == frame
        v1 = self.sim_data.loc[mask][['x', 'y']].to_numpy()[0]
        v_next = self.traj_rest[0]
        poly = get_polygons(v1, v_next, 20)

        mask = (Trajectory.full_sim_data['index'] != self.index) & (Trajectory.full_sim_data['frame'] == frame)
        all_df = Trajectory.full_sim_data.loc[mask]

        p = Point(v1)
        d_zone = []
        for z in range(len(poly)):
            zone = poly[z]
            distances = [[] for _ in range(len(poly))]
            for _, xy in all_df.iterrows():
                pz = Point([xy['x'], xy['y']])
                if zone.contains(pz):
                    distances[z].append(p.distance(pz))
            try:
                d_zone.append(min(distances[z]))
            except ValueError:
                d_zone.append(1000)

        mask = self.sim_data['frame'] == frame
        self.sim_data.loc[mask, ['z_' + str(i) for i in range(20)]] = poly
        self.sim_data.loc[mask, ['d_z_' + str(i) for i in range(20)]] = d_zone

        mask = (Trajectory.full_sim_data['frame'] == frame) & (Trajectory.full_sim_data['index'] == self.index)
        Trajectory.full_sim_data.loc[mask, ['z_' + str(i) for i in range(20)]] = poly
        Trajectory.full_sim_data.loc[mask, ['d_z_' + str(i) for i in range(20)]] = d_zone

        return self.sim_data

    def predict(self):
        return 10

    def step(self, frame: int, i: int = -1):
        """
        execute 1 simulation step and add new data to simulation data

        :param frame: int, fetch signal color
        :param i: int, which index to simulate from, use -1 to continue from previous point
        :return: DataFrame, all simulated data
        """
        if self.sim_data is None:  # if no sim_data, step() cannot be executed
            raise RuntimeError('sim_data not initialized')

        d = {}
        cols = ['index', 'x', 'y', 'd_t-1', 'd_t-2', 'd_t-3', 'd_light', 'l0', 'l1',
                'l2', 'l3', 'dir_0', 'dir_1', 'dir_2']

        # calculate distance, get xy and update traj_rest
        d_travel = self.predict()
        self.distances.append(d_travel)
        x, y, self.traj_rest = self.traverse_trajectory(self.sim_data['x'].iloc[i], self.sim_data['y'].iloc[i],
                                                        d_travel, self.traj_rest)
        d['index'] = self.index
        d['x'] = x
        d['y'] = y
        self.xy_hist.append([x, y])

        # assign previous distances from dist hist
        for n in range(1, 4):
            d['d_t-' + str(n)] = self.distances[i-n+1]

        # calculate distance from xy to light midpoint
        d['d_light'] = self.distance(x, y, self.light_mid[0], self.light_mid[1])

        # one-hot-encode current light signal
        l_color = self.l_df.loc[frame][str(self.light_index)]
        encoding = self.l_enc.transform([[l_color]]).toarray()
        for n in range(4):
            d['l' + str(n)] = encoding[0, n]

        # assign direction
        for n in ['dir_0', 'dir_1', 'dir_2']:
            d[n] = [self.data[n][0]]

        # updated frame
        d['frame'] = frame
        d['class'] = self.c
        self.sim_data = pd.concat([self.sim_data, pd.DataFrame(d)], ignore_index=True)
        Trajectory.full_sim_data = pd.concat((Trajectory.full_sim_data, pd.DataFrame(d)))
        return self.sim_data

    @staticmethod
    def traverse_trajectory(x_t: float, y_t: float, d_travel: float, traj):
        """
        method for traversing along trajectory, travels d_travel distance along given trajectory traj

        :param x_t: int, starting x-coordinate
        :param y_t: int, starting y-coordinate
        :param d_travel: int, distance to travel
        :param traj: trajectory to traverse
        :return: new x, y and the remaining trajectory
        """
        if not len(traj):
            return x_t, y_t, traj
        d_to_traj = Trajectory.distance(x_t, y_t, traj[0, 0], traj[0, 1])
        if d_travel <= d_to_traj:
            v = (traj[0] - np.array([x_t, y_t])) / Trajectory.distance(x_t, y_t, traj[0, 0], traj[0, 1])
            x_t += v[0] * d_travel
            y_t += v[1] * d_travel
            return x_t, y_t, traj
        return Trajectory.traverse_trajectory(traj[0, 0], traj[0, 1], (d_travel - d_to_traj), traj[1:])

    @staticmethod
    def distance(x_0: float, y_0: float, x_1: float, y_1: float):
        """
        calculate euclidean distance between points
        """
        return np.linalg.norm(np.array([x_0, y_0]) - np.array([x_1, y_1]))

    @staticmethod
    def reset_full_sim_data():
        Trajectory.full_sim_data = pd.DataFrame(columns=Trajectory.full_sim_data_cols)


if __name__ == '__main__':
    # nndf = Wrangler.load_pickle('data/nndf.pkl')
    pdf = Wrangler.load_pickle('data/pdf_zones.pkl')
    l_xy = Wrangler.load_pickle('bsc-3m/signal_lines_true.pickle')
    l_df = pd.read_csv('bsc-3m/signals_dense.csv')

    t = Trajectory(pdf.iloc[0], l_df, l_xy, 0)
    t.init_sim(0)
    print(t.sim_data)
    t.get_d_zones(0)
    print(t.sim_data)
    print(t.full_sim_data)
    # t.get_d_zones(0)
