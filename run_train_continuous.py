# run_train_continuous.py
# 示例：用 ES 训练连续策略（4-d action）控制整个编队（单智能体）
import os
import yaml
import numpy as np

from env.core import UAVEnv
from env import scenes
from policy.mlp_continuous import MLPPolicyContinuous
from policy.es import train_policy_es

if __name__ == "__main__":
    # load config if exist
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Override ES params here if required
    es_config = {
        'population_size': config.get('population_size', 60),
        'elite_size': config.get('elite_size', 12),
        'iterations': config.get('iterations', 200),
        'sigma': config.get('sigma', 0.1),
        'learning_rate': config.get('learning_rate', 0.05)
    }

    # train on a scenario (example S1)
    scen = "S2"
    scene_cfg = scenes.get_scene_config(scen)
    env_cfg = {
        'dt': 1.0,
        'blue_max_speed': 12.0,
        'red_max_speed': 9.0,
        'max_steps': config.get('max_steps', 300),
        'alpha': config.get('alpha', 2.0),
        'beta': config.get('beta', 1.5),
        'gamma': config.get('gamma', 0.5),
        'delta': config.get('delta', 0.1),
        'escape_penalty': config.get('escape_penalty', 8.0),
        'terminal_reward': config.get('terminal_reward', 50.0)
    }
    env = UAVEnv(scene_cfg, global_cfg=env_cfg)
    policy = MLPPolicyContinuous(input_dim=10, hidden_dims=[64,64], action_dim=4)
    trained_policy = train_policy_es(env, policy, es_config)
    os.makedirs("models", exist_ok=True)
    np.save(os.path.join("models", f"M_{scen}_cont.npy"), trained_policy.get_params())
    print("Training finished, model saved.")