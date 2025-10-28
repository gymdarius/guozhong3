import numpy as np

def train_policy_es(env, policy, config):
    """
    Train the given policy in the environment using Evolution Strategies (CEM/ES).
    config: dictionary containing hyperparameters:
      - population_size (int)
      - elite_size (int)
      - iterations (int)
      - sigma (float)
      - learning_rate (float)
      - eval_episodes (int)  # 新增，用于每个参数样本评估的 episode 数，降低方差
    Returns the trained policy (also modified in-place).
    """
    pop_size = int(config.get('population_size', 50))
    elite_size = int(config.get('elite_size', 10))
    iterations = int(config.get('iterations', 200))
    sigma = float(config.get('sigma', 0.1))
    lr = float(config.get('learning_rate', 0.05))
    eval_episodes = int(config.get('eval_episodes', 1))

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
            # Run eval_episodes episodes and average reward
            total_reward_avg = 0.0
            for e in range(eval_episodes):
                obs = env.reset()
                done = False
                total_reward = 0.0
                # run one episode
                while not done:
                    action = policy.sample_action(obs)
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                total_reward_avg += total_reward
            rewards[j] = total_reward_avg / eval_episodes

        # Select elites
        elite_indices = np.argsort(rewards)[-elite_size:]
        elite_noises = noises[elite_indices]
        # Compute new mean via CEM: mean of elites in param space
        elite_params = mean_params + sigma * elite_noises
        new_mean = np.mean(elite_params, axis=0)
        # Update mean parameters with learning rate (smoothing)
        mean_params = mean_params + lr * (new_mean - mean_params)

        # Logging progress
        if (iteration + 1) % max(1, iterations // 20) == 0 or iteration == 0:
            avg_reward = np.mean(rewards)
            max_reward = np.max(rewards)
            print(f"Iteration {iteration+1}/{iterations}: avg_reward={avg_reward:.2f}, max_reward={max_reward:.2f}")

    # Set policy to final parameters
    policy.set_params(mean_params)
    return policy