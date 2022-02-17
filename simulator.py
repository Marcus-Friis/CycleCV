import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


class Simulator:
    def __init__(self, model, data):
        self.data = data
        self.model = model
        self._features = self.data.shape[-1]

    def _prepare_sample(self, prev, pred):
        next_sample = np.array([
            prev[0] + pred[0],  # x_pos
            prev[1] + pred[1],  # y_pos
            pred[0],  # x_vec
            pred[1],  # y_vec
            prev[2],  # x_vec2
            prev[3],  # y_vec2
            prev[-2],  # x_dest
            prev[-1]  # y_dest
        ])
        return next_sample.reshape(-1, self._features)

    def _simulate_sample(self, sample, iterations=100):
        sample = sample.reshape(-1, self._features)
        for i in range(iterations):
            pred = self.model.predict(sample[-1:])[0]
            new_sample = self._prepare_sample(sample[-1], pred)
            sample = np.concatenate((sample, new_sample))
        return sample

    def simulate(self, iterations=100):
        # trajs = np.array([[]]).reshape(0, self._features)
        trajs = []
        for sample in self.data:
            traj = self._simulate_sample(sample, iterations=iterations)
            # trajs = np.concatenate((trajs, traj))
            trajs.append(traj)
        return np.array(trajs)

    @staticmethod
    def plot_simulation(trajs):
        fig, ax = plt.subplots(figsize=(20, 20))
        im = Image.open("intersection2.png")
        im = ImageOps.flip(im)
        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 720)
        ax.imshow(im, origin='lower')

        for traj in trajs:
            ax.scatter(traj[:, 0], traj[:, 1])

        return ax


if __name__ == '__main__':
    import pickle

    np.set_printoptions(precision=4, suppress=True)

    with open("train.pkl", "rb") as f:
        df = pickle.load(f)

    x = df[['x_pos', 'y_pos', 'x_vec', 'y_vec', 'x_vec2', 'y_vec2', 'x_dest', 'y_dest']].to_numpy()
    y = df[['x_tar', 'y_tar']].to_numpy()

    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)

    sim = Simulator(clf, x[:2])

    print(sim.simulate().shape)
