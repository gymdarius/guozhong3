# policy/mlp_continuous.py
# 简单的 MLP 连续动作策略（输出均值与可学习的 log_std）
import numpy as np

class MLPPolicyContinuous:
    """MLP policy that outputs continuous action means and a small learnable std."""
    def __init__(self, input_dim=10, hidden_dims=[64,64], action_dim=4):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        # weights initialization
        self.W1 = np.random.randn(input_dim, hidden_dims[0]) * 0.1
        self.b1 = np.zeros(hidden_dims[0])
        self.W2 = np.random.randn(hidden_dims[0], hidden_dims[1]) * 0.1
        self.b2 = np.zeros(hidden_dims[1])
        self.W3 = np.random.randn(hidden_dims[1], action_dim) * 0.1
        self.b3 = np.zeros(action_dim)
        # log std (learnable)
        self.log_std = np.ones(action_dim) * -1.0

    def forward(self, obs):
        x = obs
        z1 = x.dot(self.W1) + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1.dot(self.W2) + self.b2
        a2 = np.maximum(z2, 0)
        mean = a2.dot(self.W3) + self.b3
        std = np.exp(self.log_std)
        return mean, std

    def sample_action(self, obs):
        mean, std = self.forward(obs)
        action = mean + np.random.randn(*mean.shape) * std
        return action

    def predict(self, obs):
        mean, std = self.forward(obs)
        return mean.astype(float)

    def get_params(self):
        return np.concatenate([self.W1.ravel(), self.b1.ravel(),
                               self.W2.ravel(), self.b2.ravel(),
                               self.W3.ravel(), self.b3.ravel(),
                               self.log_std.ravel()])

    def set_params(self, params):
        # determine sizes
        w1_size = self.input_dim * self.hidden_dims[0]
        b1_size = self.hidden_dims[0]
        w2_size = self.hidden_dims[0] * self.hidden_dims[1]
        b2_size = self.hidden_dims[1]
        w3_size = self.hidden_dims[1] * self.action_dim
        b3_size = self.action_dim
        idx = 0
        self.W1 = params[idx: idx + w1_size].reshape(self.input_dim, self.hidden_dims[0]); idx += w1_size
        self.b1 = params[idx: idx + b1_size].reshape(self.hidden_dims[0],); idx += b1_size
        self.W2 = params[idx: idx + w2_size].reshape(self.hidden_dims[0], self.hidden_dims[1]); idx += w2_size
        self.b2 = params[idx: idx + b2_size].reshape(self.hidden_dims[1],); idx += b2_size
        self.W3 = params[idx: idx + w3_size].reshape(self.hidden_dims[1], self.action_dim); idx += w3_size
        self.b3 = params[idx: idx + b3_size].reshape(self.action_dim,); idx += b3_size
        self.log_std = params[idx: idx + self.action_dim].reshape(self.action_dim,)