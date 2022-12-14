import itertools
from random import randint, seed

import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import seaborn as sns
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers

from LSTM import LSTM
from pytrajectory import LSTMTrajectory
from trajectory import Trajectory
from wrangler import Wrangler
from sktrajectory import SKTrajectory

import warnings
warnings.filterwarnings('ignore')


def animate_simulate(traj_count: int, reg, path: str):
    # init figure
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 720)
    # load and show image on figure
    img = Image.open("intersection2.png")
    img = ImageOps.flip(img)
    im = ax.imshow(img, origin='lower')

    # init moving elements
    ln = [ax.plot([], [])[0] for _ in range(l_df.shape[-1] - 1)]  # traffic light lines
    txt = ax.text(20, 20, '', fontsize=35, color='w')  # frame number
    sc = ax.scatter([], [], c='r', s=200)  # cars
    patches = ln + [sc] + [txt]

    # init trajectories
    Trajectory.index_gen = itertools.count()  # reset index generator

    # depending on model, use pytorch LSTM interface or sklearn
    # t = [LSTMTrajectory(pdf.iloc[randint(0, n_trajectories)], l_df, l_xy, reg) for i in range(traj_count)]
    t = [SKTrajectory(pdf.iloc[randint(0, n_trajectories)], l_df, l_xy, reg) for i in range(traj_count)]

    t[0].reset_full_sim_data()  # reset full_sim_data
    start = randint(0, 10000)  # randomly generate start frame
    for i in range(traj_count):
        t[i].init_sim(start)  # init all trajectories
    for i in range(traj_count):
        t[i].get_d_zones(start)  # get zones for each trajectory

    # init function, inits traffic light lines
    def init():
        for i in range(len(ln)):
            ln[i].set_data(l_xy[i]['x'], l_xy[i]['y'])

        return patches

    # update function, specify behaviour for each frame in movie
    def update(frame):
        # update traffic lights
        row = l_df.loc[frame]  # get light status to current frame
        for i in range(l_df.shape[1] - 1):
            ln[i].set_color(['red', 'orange', 'yellow', 'green'][row[str(i)]])  # update each light

        # update frame number display
        txt.set(text=str(frame))

        # update each trajectory
        for i in range(traj_count):
            if len(t[i].traj_rest) > 0:
                t[i].step(frame)

        # update each trajectory's zone
        for i in range(traj_count):
            if len(t[i].traj_rest) > 0:
                t[i].get_d_zones(frame)

        # plot all dots
        mask = t[0].full_sim_data['frame'] == frame  # get all vehicles in current frame
        xy = t[0].full_sim_data.loc[mask, ['x', 'y']]  # get xy for all vehicles to current frame
        sc.set_offsets(xy)  # plot
        # sc.set(color=[sns.color_palette()[i] for i in t[0].full_sim_data.loc[mask, 'index']])  # update colors

        return patches

    ani = FuncAnimation(fig, update, frames=np.arange(start + 5, start + 1000, 5), init_func=init)

    Writer = writers['ffmpeg']
    writer = Writer(fps=6)  # , bitrate=1800)
    ani.save(path, writer)


if __name__ == '__main__':
    pdf = Wrangler.load_pickle('data/pdf_test.pkl')
    n_trajectories = pdf.shape[0]

    l_df = pd.read_csv('bsc-3m/signals_dense.csv')
    l_xy = Wrangler.load_pickle('bsc-3m/signal_lines_true.pickle')

    # load model ------------------------------------------
    # if LSTM
    # model = LSTM(33, 100, 1, 2, 0.2, 849)
    # model.load_state_dict(torch.load('models/lstm.pt'))
    # model.eval()
    # if sklearn interface
    # reg = Wrangler.load_pickle('models/xgb.pkl')
    # reg = Wrangler.load_pickle('models/lgbmr.pkl')
    # reg = Wrangler.load_pickle('models/cgb.pkl')
    # -----------------------------------------------------
    # set path to writer, this uses ffmpeg
    mpl.rcParams['animation.ffmpeg_path'] = r'D:\ffmpeg\bin\ffmpeg.exe'

    regressors = [Wrangler.load_pickle(n) for n in ['models/xgb.pkl', 'models/lgbmr.pkl',
                                                    'models/cgb.pkl', 'models/mlp.pkl']]
    regressor_names = ['xgb', 'lgbmr', 'cgb', 'mlp']

    num_movies = 30  # number of videos to produce
    for reg_index, reg in enumerate(regressors):
        seed(1)  # set seed, ensures fair comparison of models
        for movie in range(num_movies):
            print(f'creating\t{movie}...')
            traj_count = randint(2, 10)
            animate_simulate(traj_count, reg, f'movies/{regressor_names[reg_index]}/sim_{movie}.mp4')
