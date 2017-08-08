

class Dynamics(object):

    def __init__(self):
        self.Fm = None
        self.fv = None
        self.dyn_cov = None

    def fit(self, X, U):
        # TODO: Implement fit
        return self.Fm, self.fv, self.dyn_cov