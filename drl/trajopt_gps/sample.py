import numpy as np


class Sample(object):

    def __init__(self, T, dX, dU):
        self.T = T
        self.dX = dX
        self.dU = dU
        self.X = np.zeros((T+1, dX))
        self.U = np.zeros((T, dU))
        self.count = 0

    def add(self, x, u=None):
        assert len(x.shape) > 1
        self.X[self.count:self.count+x.shape[0], :] = x
        if u is not None:
            assert x.shape[0] == u.shape[0]
            self.U[self.count:self.count+x.shape[0], :] = u
        self.count += x.shape[0]

    def get_X(self):
        return self.X

    def get_U(self):
        return self.U


class SampleList(object):

    def __init__(self):
        self.T = None
        self.dX = None
        self.dU = None
        self._samples = []

    def add(self, samples):
        if type(samples) is Sample:
            samples = [samples]

        self._set_vars(samples)
        self._samples += samples

    def _set_vars(self, samples):
        if self.T is None and self.dX is None and self.dU is None:
            self.T = samples[0].T
            self.dX = samples[0].dX
            self.dU = samples[0].dU

        for sample in samples:
            assert sample.T == self.T
            assert sample.dX == self.dX
            assert sample.dU == self.dU

    def get_X(self, idx=None):
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_X() for i in idx])

    def get_U(self, idx=None):
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_U() for i in idx])

    def get_samples(self, idx=None):
        if idx is None:
            idx = range(len(self._samples))
        elif type(idx) == int:
            return self._samples[idx]
        return [self._samples[i] for i in idx]

    def len(self):
        return len(self._samples)