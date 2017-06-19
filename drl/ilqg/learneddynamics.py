import numpy as np
from sklearn.linear_model import LinearRegression


class LearnedDynamics(object):
    """
    Learned Dynamics object is used for fitting time-varying linear models.
    For each time-step 't' a linear model is fitted using the data of time-step 't - N' till 't + N'.

    Linear models:

        Y = X*beta' + epsilon

        with:

        Y = x(k+1)             - data matrix for outputs
        X = [x(k) u(k)]        - data matrix for features
        fx = beta[:state_dim]  - derivative wrt state
        fu = beta[state_dim:]  - derivative wrt control input
    """

    def __init__(self, max_steps, num_episodes, obs_dim, action_dim, N=1):
        """
        Constructs a new 'LearnedDynamics' object.

        :param max_steps: maximum number of linear models
        :param num_episodes: maximum number of samples for each model
        :param obs_dim: state dimension
        :param action_dim: action dimension
        :param N: number of data samples to use for fitting the model
        """
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.N = N
        self.reset()

    def reset(self):
        """
        Resets the data matrices, models and current_step parameter.
        """
        self.X = np.zeros((self.max_steps, self.num_episodes, self.obs_dim + self.action_dim))
        self.Y = np.zeros((self.max_steps, self.num_episodes, self.obs_dim))
        self.cur_step = 0
        self.models = []
        for _ in range(self.max_steps):
            self.models.append(LinearRegression())

    def add(self, episode, step, x, u, x_new):
        """
        Add new data sample to data matrices.

        :param episode:
        :param step:
        :param x:
        :param u:
        :param x_new:
        :return:
        """
        self.X[step, episode, :] = np.concatenate((x, u))
        self.Y[step, episode, :] = x_new

    def fit(self, sample_weight=None):
        """
        Refits the linear models according to the data in data matrices X and Y.

        :param sample_weight: weight on each sample based on episode reward (NOT IMPLEMENTED)
        """
        for i in range(self.N):
            x_tmp = self.X[:i + self.N, :, :]
            x_tmp = np.reshape(x_tmp, [x_tmp.shape[0] * x_tmp.shape[1], x_tmp.shape[2]])
            y_tmp = self.Y[:i + self.N, :, :]
            y_tmp = np.reshape(y_tmp, [y_tmp.shape[0] * y_tmp.shape[1], y_tmp.shape[2]])
            self.models[i].fit(x_tmp, y_tmp)

        for i in range(self.N, self.max_steps - self.N):
            x_tmp = self.X[i - self.N: i + self.N, :, :]
            x_tmp = np.reshape(x_tmp, [x_tmp.shape[0] * x_tmp.shape[1], x_tmp.shape[2]])
            y_tmp = self.Y[i - self.N: i + self.N, :, :]
            y_tmp = np.reshape(y_tmp, [y_tmp.shape[0] * y_tmp.shape[1], y_tmp.shape[2]])
            self.models[i].fit(x_tmp, y_tmp)

        for i in range(self.max_steps - self.N, self.max_steps):
            x_tmp = self.X[i:, :, :]
            x_tmp = np.reshape(x_tmp, [x_tmp.shape[0] * x_tmp.shape[1], x_tmp.shape[2]])
            y_tmp = self.Y[i:, :, :]
            y_tmp = np.reshape(y_tmp, [y_tmp.shape[0] * y_tmp.shape[1], y_tmp.shape[2]])
            self.models[i].fit(x_tmp, y_tmp)

    def set_cur_step(self, step):
        """
        Set the current_step parameter.
        Current_step parameter ensures the right model is used for the iLQG roll-out.


        :param step: current step
        """
        self.cur_step = step

    def dynamics_func(self, x, u):
        """
        Predicts the next state and derivatives according the linear model.
        The linear model corresponding to current_step parameter is used.

        :param x: state
        :param u: control input
        :return: next state (f), fx (state-derivative), fu (input-derivative), fxx (=None), fuu (=None), fxu (=None)
        """
        u[np.isnan(u)] = 0.
        X_in = np.concatenate((x, u)).reshape(1, -1)
        beta = self.models[self.cur_step].coef_
        return self.models[self.cur_step].predict(X_in)[0], beta[:, :self.obs_dim].T, beta[:, self.obs_dim:].T, None, None, None