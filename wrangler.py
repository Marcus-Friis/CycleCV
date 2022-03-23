import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


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
        self.light_dict = light_dict
        if light_dict is None:
            self.light_dict = {0: 7, 1: 6, 2: 5, 3: 9, 4: 8, 5: 6, 6: 10, 7: 6, 8: 4, 9: 4, 10: 11, 11: 5}

        # set mapping of cluster : direction, use default if None
        self.direction_dict = direction_dict
        if direction_dict is None:
            self.direction_dict = {
                0: 'right', 1: 'straight', 2: 'left', 3: 'left', 4: 'straight', 5: 'left',
                6: 'right', 7: 'right', 8: 'straight', 9: 'left', 10: 'right', 11: 'straight'
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

    def init_attributes(self, step_size: int = 1, dump: bool = False, path: str = None):
        """
        wrangle data and calculate various attributes from DataFrame,
        must be done before other wrangling,
        can also be loaded with load_pdf

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
            'class': [],
            'light_color': [],
            'direction': []

        }

        l_enc = OneHotEncoder(handle_unknown='ignore').fit(np.array([0, 1, 2, 3]).reshape(-1, 1))
        d_enc = OneHotEncoder(handle_unknown='ignore').fit(np.array(['left', 'straight', 'right']).reshape(-1, 1))
        classes = np.unique(self.df['class']).reshape(-1, 1)
        c_enc = OneHotEncoder(handle_unknown='ignore').fit(classes)

        for i, n in enumerate(classes):
            d['c_' + str(i)] = []

        for _, row in self.df.iterrows():
            rowx = np.array(row['xs'][::step_size])
            rowy = np.array(row['ys'][::step_size])
            frames = np.array(row['frames'][::step_size][:-1])
            l_mid = self._get_mid(self.l_xy[self.light_dict[row['cluster']]])

            encoding = c_enc.transform([[row['class']]]).toarray()
            for i in range(classes.shape[0]):
                d['c_'+str(i)].append(encoding[0,i])

            d['x'].append(rowx[:-1])
            d['y'].append(rowy[:-1])
            l_color = np.array([self.l_df.loc[f][str(self.light_dict[row['cluster']])] for f in frames])
            d['light_color'].append(l_color)
            d['d_light'].append(self._d2l(rowx[:-1], rowy[:-1], l_mid))
            encoding = l_enc.transform(l_color.reshape(-1, 1)).toarray()
            for n in range(4):
                d['l' + str(n)].append(encoding[:, n])

            eucs = self._euc(rowx, rowy)
            d['euc'].append(eucs)
            d['frames'].append(frames)
            d['class'].append(row['class'])
            d['cluster'].append(row['cluster'])
            direction = self.direction_dict[row['cluster']]
            d['direction'].append(direction)
            encoding = d_enc.transform([[direction]]).toarray()
            for n in range(3):
                d['dir_' + str(n)].append([encoding[0, n]]*len(rowx-1))

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

        self.pdf = pd.DataFrame(d)

        if dump:
            if path is None:
                self.dump_pickle(self.pdf, 'pdf.pkl')
            else:
                self.dump_pickle(self.pdf, path)

        return self

    def load_pdf(self, path: str):
        """
        load pdf instead of running init_attributes

        :param path: path to pdf
        :return:
        """
        self.pdf = self.load_pickle(path)
        return self

    def get_nndf(self, dump: bool = False, path: str = None):
        """
        get data formatted for training neural networks,
        must run init_attributes or load_pdf before this method

        :param dump: save file
        :param path: path to file
        :return: nndf
        """
        if self.pdf is None:
            raise RuntimeError('Must init pdf with init_attributes or load_pdf first')

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

        for index, row in self.pdf.iterrows():
            for i in range(3, len(row['x'])):
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

        self.nndf = pd.DataFrame(d)

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


def main():
    df = Wrangler.load_pickle('bsc-3m/traj_clustered.pkl')
    frames = Wrangler.load_pickle('bsc-3m/traj_01_elab_new.pkl')
    df = df.join(frames['frames'])
    df = df.loc[df['cluster'] != -1]

    l_xy = Wrangler.load_pickle('bsc-3m/signal_lines_true.pickle')
    l_df = pd.read_csv('bsc-3m/signals_dense.csv')

    wng = Wrangler(df, l_xy, l_df).init_attributes(20).get_nndf()
    a = wng.nndf
    print(a)


if __name__ == '__main__':
    main()
