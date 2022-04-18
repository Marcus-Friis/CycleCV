import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from matplotlib.animation import FuncAnimation, writers

from wrangler import Wrangler

# must run vide_to_frames.py before this script
if __name__ == '__main__':
    # if all_df is generated, run below code
    all_df = Wrangler.load_pickle('data/all_df.pkl')
    # -----------------------------------------------
    # if all_df is not generated, run below code
    # df = Wrangler.load_pickle('bsc-3m/traj_01_elab.pkl')
    # df_frames = Wrangler.load_pickle('bsc-3m/traj_01_elab_new.pkl')
    # df = df.join(df_frames['frames'])
    # all_df = Wrangler.get_all_df(df)

    # load signals data
    l_df = pd.read_csv('bsc-3m/signals_dense.csv')
    l_xy = Wrangler.load_pickle('bsc-3m/signal_lines_true.pickle')

    # dict for color coding objects
    class_color = {
        'Bicycle': 'r',
        'Bus': 'b',
        'Car': 'b',
        'Heavy Vehicle': 'b',
        'Light Truck': 'b',
        'Motorcycle': 'b',
        'Pedestrian': 'g',
        'Van': 'b'
    }

    # create figure
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 720)
    img = Image.open("intersection2.png")
    img = ImageOps.flip(img)

    # create patches, to be animated objects
    im = ax.imshow(img, origin='lower')
    ln = [ax.plot([], [])[0] for _ in range(l_df.shape[-1] - 1)]
    txt = ax.text(20, 20, '', fontsize=35, color='w')
    sc = ax.scatter([], [], s=200)
    patches = ln + [sc] + [txt] + [im]

    # animation init func, initializes all light lines
    def init():
        for i in range(len(ln)):
            ln[i].set_data(l_xy[i]['x'], l_xy[i]['y'])
        return patches

    # animation update func, run this func to each frame
    def update(frame):
        # update all light lines to current light color
        row = l_df.loc[frame]
        for i in range(l_df.shape[1] - 1):
            ln[i].set_color(['red', 'orange', 'yellow', 'green'][row[str(i)]])

        # update frame counter
        txt.set(text=str(frame))

        # update all current object positions and colors
        mask = all_df['frame'] == frame
        xy = all_df.loc[mask][['x', 'y']].to_numpy()
        sc.set_offsets(xy)
        sc.set(color=[class_color[row['class']] for _, row in all_df.loc[mask].iterrows()])

        # update current image
        img = Image.open('frames/frame' + str(frame) + '.jpg')  # + pic_frames[frame])
        img = ImageOps.flip(img)
        im.set_array(img)

        return patches

    # create animation
    mpl.rcParams['animation.ffmpeg_path'] = r'D:\ffmpeg\bin\ffmpeg.exe'
    ani = FuncAnimation(fig, update, frames=np.arange(0, 3000),  # run animation from frame 0 to frame 3000
                        init_func=init)

    Writer = writers['ffmpeg']
    writer = Writer(fps=30)
    ani.save('data_animation.mp4', writer)
