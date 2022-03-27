import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from abc import ABC, abstractmethod


class Trajectory(ABC):
    full_sim_data = pd.DataFrame(columns=['x', 'y', 'd_t-1', 'd_t-2', 'd_t-3', 'd_light', 'l0', 'l1',
                                          'l2', 'l3', 'dir_0', 'dir_1', 'dir_2', 'frame', 'class'])
    l_enc = OneHotEncoder(handle_unknown='ignore').fit(np.array([[0, 1, 2, 3]]).reshape(-1, 1))

    def __init__(self, data, l_df, l_xy, clf):
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
        self.clf = clf

        # trajectory variables for full trajectory and remaining when simulating
        self.traj_full = np.array([data['x'], data['y']]).T
        self.traj_rest = self.traj_full[1:]

        # variables for getting relevant light info
        self.light_index = int(self.data['light_index'])
        n = self.l_xy[self.light_index]
        self.light_mid = np.array([sum(n['x']) / len(n['x']), sum(n['y']) / len(n['y'])])

        # the simulated data and history of distances
        self.sim_data = None
        self.distances = []
        self.c = self.data['class']

    def init_sim(self, frame: int, i: int = 0):
        """
        initialize simulation and format data

        :param frame: int, for getting signal color
        :param i: int, which index to start from
        :return: DataFrame, ready to simulate
        """
        d = {'x': self.data['x'][i], 'y': self.data['y'][i]}  # init xy coordinate

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

        # save data, add to DataFrame of all data
        self.sim_data = pd.DataFrame(d)
        Trajectory.full_sim_data = pd.concat((Trajectory.full_sim_data, self.sim_data))
        return self.sim_data

    @abstractmethod
    def predict(self):
        pass

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
        cols = ['x', 'y', 'd_t-1', 'd_t-2', 'd_t-3', 'd_light', 'l0', 'l1',
                'l2', 'l3', 'dir_0', 'dir_1', 'dir_2']

        # calculate distance, get xy and update traj_rest
        d_travel = self.predict()
        self.distances.append(d_travel)
        x, y, self.traj_rest = self.traverse_trajectory(self.sim_data['x'].iloc[i], self.sim_data['y'].iloc[i],
                                                        d_travel, self.traj_rest)
        d['x'] = x
        d['y'] = y

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


def main():
    pass


if __name__ == '__main__':
    main()