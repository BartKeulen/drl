from replaybuffer import ReplayBuffer
import numpy as np


class ReplayBufferStat(ReplayBuffer):

    def __init__(self, buffer_size, s_bounds, a_bounds, C, random_seed=123):
        super(ReplayBufferStat, self).__init__(buffer_size, random_seed)
        self.s_bounds = s_bounds
        self.a_bounds = a_bounds
        self.C = C

    def calc_kernel_density(self, states, actions):
        s_buffer = np.array([_[0] for _ in self.buffer])
        a_buffer = np.array([_[1] for _ in self.buffer])

        buffer = np.concatenate((s_buffer, a_buffer), axis=1)
        values = np.concatenate((states, actions), axis=1)
        densities = []
        for i in range(values.shape[0]):
            delta = buffer - values[i, :]
            squared = np.power(delta, 2) / (np.concatenate([self.s_bounds, self.a_bounds]) * 2) * self.C
            density = np.exp(-np.sum(squared, axis=1))
            densities.append(np.sum(density) / self.count)

        return np.array(densities)

def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    buffer = ReplayBufferStat(500, 5., 2., 5.)
    C = 5

    for i in xrange(250):
        buffer.add((np.random.rand(1)-0.5)*10, (np.random.rand(1)-0.5)*4, np.random.randn(1), False, np.random.randn(1)*5)

    for i in xrange(250):
        buffer.add((np.random.rand(1))*2.5 + 3, np.random.rand(1)*1 + 1.5, np.random.randn(1), False, np.random.randn(1)*5)

    s_buffer = np.array([_[0] for _ in buffer.buffer])
    a_buffer = np.array([_[1] for _ in buffer.buffer])

    s_max = 5.
    a_max = 2.

    resolution = 100
    s = np.linspace(-s_max, s_max, resolution)
    a = np.linspace(-a_max, a_max, resolution)

    plt.figure()
    plt.scatter(s_buffer, a_buffer)

    sv, av = np.meshgrid(s, a, sparse=False, indexing='ij')
    svv = np.zeros(resolution*resolution)
    avv = np.zeros(resolution*resolution)
    count = 0
    for i in range(resolution):
        for j in range(resolution):
            svv[count] = sv[i, j]
            avv[count] = av[i, j]
            count += 1

    svv = np.transpose(svv[np.newaxis])
    avv = np.transpose(avv[np.newaxis])

    values = buffer.calc_kernel_density(svv, avv)
    values = np.reshape(values, (resolution, resolution))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(sv, av, values, cmap=cm.coolwarm)

    plt.show()

if __name__ == "__main__":
    main()