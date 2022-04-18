import pandas as pd
import torch
import torch.nn as nn

from trajectory import Trajectory
from wrangler import Wrangler


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


class LSTMTrajectory(Trajectory):
    def __init__(self, data, l_df, l_xy, clf):
        super(LSTMTrajectory, self).__init__(data, l_df, l_xy)
        self.clf = clf

    def predict(self):
        """
        overrides method of superclass, used for predicting distance.
        PyTrajectory uses pytorch interface.

        :return: float, euclidean distance
        """
        with torch.no_grad():
            cols = ['x', 'y', 'd_t-1', 'd_t-2', 'd_t-3', 'd_light', 'l0', 'l1',
                    'l2', 'l3', 'dir_0', 'dir_1', 'dir_2'] + ['d_z_' + str(i) for i in range(20)]
            x = self.sim_data[cols].to_numpy()
            xx = torch.from_numpy(x).reshape(1, -1, len(cols)).float()
            lens = torch.tensor([xx.size(1)])
            out = self.clf(xx, lens)
        return float(out[-1, lens[0] - 1, 0])


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
