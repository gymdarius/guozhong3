import numpy as np

def train_policy_es(env, policy, config):
    """
    Train the given policy in the environment using Evolution Strategies (CEM/ES).
    config: dictionary containing hyperparameters (population_size, elite_size, iterations, sigma, learning_rate).
    Returns the trained policy (also modified in-place).
    """
    pop_size = config.get('population_size', 50)
    elite_size = config.get('elite_size', 10)
    iterations = config.get('iterations', 200)
    sigma = config.get('sigma', 0.1)
    lr = config.get('learning_rate', 0.05)
    # Get initial parameter vector
    params = policy.get_params()
    mean_params = params.copy()
    param_dim = len(params)
    for iteration in range(iterations):
        # Sample population of parameters
        noises = np.random.randn(pop_size, param_dim)
        rewards = np.zeros(pop_size)
        # Evaluate each sample
        for j in range(pop_size):
            sample_params = mean_params + sigma * noises[j]
            policy.set_params(sample_params)
            # Run one episode to get reward
            total_reward = 0.0
            obs = env.reset()
            done = False
            while not done:
                action = policy.sample_action(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
            rewards[j] = total_reward
        # Select elites
        elite_indices = np.argsort(rewards)[-elite_size:]
        elite_noises = noises[elite_indices]
        # Compute new mean via CEM: mean of elites
        elite_params = mean_params + sigma * elite_noises
        new_mean = np.mean(elite_params, axis=0)
        # Update mean parameters with learning rate (smoothing)
        mean_params = mean_params + lr * (new_mean - mean_params)
        # Optionally: print progress every 20 iterations
        if (iteration + 1) % 20 == 0:
            avg_reward = np.mean(rewards)
            max_reward = np.max(rewards)
            print(f"Iteration {iteration+1}/{iterations}: avg_reward={avg_reward:.2f}, max_reward={max_reward:.2f}")
    # Set policy to final parameters
    policy.set_params(mean_params)
    return policy
