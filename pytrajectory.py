import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder

from wrangler import Wrangler
from trajectory import Trajectory


class PyTrajectory(Trajectory):
    def __init__(self, data, l_df, l_xy, clf):
        super(PyTrajectory, self).__init__(data, l_df, l_xy)
        self.clf = clf

    def predict(self):
        """
        overrides method of superclass, used for predicting distance.
        PyTrajectory uses pytorch interface.

        :return: float, euclidean distance
        """
        cols = ['x', 'y', 'd_t-1', 'd_t-2', 'd_t-3', 'd_light', 'l0', 'l1',
                'l2', 'l3', 'dir_0', 'dir_1', 'dir_2']
        x_predict = torch.from_numpy(self.sim_data[cols].iloc[-1].to_numpy().reshape(1, -1))
        d_travel = float(self.clf(x_predict)[0, 0])
        return d_travel


def main():
    model = nn.Sequential(nn.Linear(13, 50), nn.ReLU(), nn.Linear(50, 1)).double()

    # nndf = Wrangler.load_pickle('data/nndf.pkl')
    pdf = Wrangler.load_pickle('data/pdf.pkl')
    l_df = pd.read_csv('bsc-3m/signals_dense.csv')
    l_xy = Wrangler.load_pickle('bsc-3m/signal_lines_true.pickle')

    t1 = PyTrajectory(pdf.iloc[0], l_df, l_xy, model)
    t2 = PyTrajectory(pdf.iloc[1], l_df, l_xy, model)
    t1.init_sim(0, 0)
    t2.init_sim(0, 0)
    for frame in range(10, 100, 10):
        t1.step(frame)
        t2.step(frame)
    print(t1.full_sim_data)
    print(t2.full_sim_data)


if __name__ == '__main__':
    main()
