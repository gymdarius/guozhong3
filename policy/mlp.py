import numpy as np

class MLPPolicy:
    """A simple Multi-Layer Perceptron policy network for discrete action selection."""
    def __init__(self, input_dim=10, hidden_dims=[64, 64], output_dim=8):
        # Initialize weight matrices and bias vectors
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        # Layer 1 weights and bias
        self.W1 = np.random.randn(input_dim, hidden_dims[0]) * 0.1
        self.b1 = np.zeros(hidden_dims[0])
        # Layer 2
        self.W2 = np.random.randn(hidden_dims[0], hidden_dims[1]) * 0.1
        self.b2 = np.zeros(hidden_dims[1])
        # Layer 3 (output layer)
        self.W3 = np.random.randn(hidden_dims[1], output_dim) * 0.1
        self.b3 = np.zeros(output_dim)
    
    def forward(self, obs):
        """Compute action probabilities from an observation vector."""
        # obs is assumed to be a 1D numpy array of length input_dim
        x = obs
        z1 = x.dot(self.W1) + self.b1
        a1 = np.maximum(z1, 0)  # ReLU activation
        z2 = a1.dot(self.W2) + self.b2
        a2 = np.maximum(z2, 0)
        z3 = a2.dot(self.W3) + self.b3
        # Softmax output
        exp = np.exp(z3 - np.max(z3))
        probs = exp / np.sum(exp)
        return probs
    
    def sample_action(self, obs):
        """Sample an action index according to the policy's output distribution."""
        probs = self.forward(obs)
        action = np.random.choice(len(probs), p=probs)
        return action
    
    def predict(self, obs):
        """Select action with highest probability (greedy action)."""
        probs = self.forward(obs)
        action = int(np.argmax(probs))
        return action
    
    def get_params(self):
        """Get a flat array of all network parameters."""
        return np.concatenate([self.W1.ravel(), self.b1.ravel(),
                               self.W2.ravel(), self.b2.ravel(),
                               self.W3.ravel(), self.b3.ravel()])
    
    def set_params(self, params):
        """Set network parameters from a flat array (must match shapes)."""
        # Determine shapes and sizes
        w1_size = self.input_dim * self.hidden_dims[0]
        b1_size = self.hidden_dims[0]
        w2_size = self.hidden_dims[0] * self.hidden_dims[1]
        b2_size = self.hidden_dims[1]
        w3_size = self.hidden_dims[1] * self.output_dim
        b3_size = self.output_dim
        # Slice the flat params accordingly
        self.W1 = params[0:w1_size].reshape(self.input_dim, self.hidden_dims[0])
        idx = w1_size
        self.b1 = params[idx: idx + b1_size].reshape(self.hidden_dims[0],)
        idx += b1_size
        self.W2 = params[idx: idx + w2_size].reshape(self.hidden_dims[0], self.hidden_dims[1])
        idx += w2_size
        self.b2 = params[idx: idx + b2_size].reshape(self.hidden_dims[1],)
        idx += b2_size
        self.W3 = params[idx: idx + w3_size].reshape(self.hidden_dims[1], self.output_dim)
        idx += w3_size
        self.b3 = params[idx: idx + b3_size].reshape(self.output_dim,)
