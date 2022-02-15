import numpy as np


class Sampler:
    def __init__(self, default=True, random_state=None):
        np.random.seed(random_state)
        self.origins = np.array([])
        self.destinations = np.array([])
        if default:
            self.origins = np.array([
                (1000, 220),    # bot right corner, straight lane
                (1080, 260),    # bot right corner, right lane
                (150, 444),     # left, straight lane
                (145, 395),     # left, right lane
                (450, 585),     # top
                (947, 508)      # right
            ])
            self.destinations = np.array([
                (490, 580),     # top
                (720, 160),     # bot
                (1000, 480),    # right
                (180, 512),     # left
            ])

    def get_sample(self, origin=None, destination=None, noise=True):
        if origin is None:
            i = np.random.randint(self.origins.shape[0])
            origin = self.origins[i]
        if destination is None:
            i = np.random.randint(self.destinations.shape[0])
            destination = self.destinations[i]
        return origin, destination

    def add_origin(self, origin):
        self.origins = np.concatenate((self.origins, origin))

    def add_destination(self, destination):
        self.destinations = np.concatenate((self.origins, destination))


if __name__ == '__main__':
    s = Sampler()
    print(s.get_sample())