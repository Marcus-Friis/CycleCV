import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder

from wrangler import Wrangler
from trajectory import Trajectory
from pytrajectory import PyTrajectory


class SKTrajectory(Trajectory):
    def predict(self):
        """
        overrides method of superclass, used for predicting distance.
        SKTrajectory uses sklearn interface.

        :return: float, euclidean distance
        """
        cols = ['x', 'y', 'd_t-1', 'd_t-2', 'd_t-3', 'd_light', 'l0', 'l1',
                'l2', 'l3', 'dir_0', 'dir_1', 'dir_2']
        x_predict = self.sim_data[cols].iloc[-1].to_numpy().reshape(1, -1)
        d_travel = self.clf.predict(x_predict)[0]
        return d_travel


def main():
    clf = Wrangler.load_pickle('models/model.pkl')
    # nndf = Wrangler.load_pickle('data/nndf.pkl')
    pdf = Wrangler.load_pickle('data/pdf.pkl')
    l_df = pd.read_csv('bsc-3m/signals_dense.csv')
    l_xy = Wrangler.load_pickle('bsc-3m/signal_lines_true.pickle')

    t1 = SKTrajectory(pdf.iloc[0], l_df, l_xy, clf)
    t2 = SKTrajectory(pdf.iloc[1], l_df, l_xy, clf)
    t1.init_sim(0, 0)
    t2.init_sim(0, 0)

    model = nn.Sequential(nn.Linear(13, 50), nn.ReLU(), nn.Linear(50, 1)).double()
    t3 = PyTrajectory(pdf.iloc[2], l_df, l_xy, model)
    t3.init_sim(0, 0)
    for frame in range(10, 100, 10):
        t1.step(frame)
        t2.step(frame)
    print(t1.full_sim_data)
    print(t2.full_sim_data)


if __name__ == '__main__':
    main()
