import argparse
import os
import numpy as np
import yaml

from env.core import UAVEnv
from env import scenes
from policy.mlp import MLPPolicy
from policy.es import train_policy_es

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UAV strategy models for specified scenario(s).")
    parser.add_argument("--scenario", type=str, default="ALL", 
                        help="Which scenario to train (S1, S2, S3, S4 or ALL for all scenarios).")
    parser.add_argument("--iterations", type=int, help="Number of training iterations (overrides config).")
    parser.add_argument("--population", type=int, help="Population size (overrides config).")
    parser.add_argument("--elite", type=int, help="Elite size (overrides config).")
    parser.add_argument("--sigma", type=float, help="Noise sigma (overrides config).")
    parser.add_argument("--lr", type=float, help="Learning rate (overrides config).")
    args = parser.parse_args()
    # Load global config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # default config if file missing
        config = {}
    # Override with command-line args if provided
    if args.iterations is not None:
        config['iterations'] = args.iterations
    if args.population is not None:
        config['population_size'] = args.population
    if args.elite is not None:
        config['elite_size'] = args.elite
    if args.sigma is not None:
        config['sigma'] = args.sigma
    if args.lr is not None:
        config['learning_rate'] = args.lr
    # Determine scenarios to train
    scenarios_to_train = []
    if args.scenario is None or args.scenario.upper() == "ALL":
        scenarios_to_train = ["S1", "S2", "S3", "S4"]
    else:
        # allow multiple scenarios separated by comma
        scen_input = args.scenario.upper()
        if scen_input.find(',') != -1:
            scenarios_to_train = [s.strip() for s in scen_input.split(',')]
        else:
            scenarios_to_train = [scen_input]
    os.makedirs("models", exist_ok=True)
    for scen in scenarios_to_train:
        print(f"=== Training model for scenario {scen} ===")
        scene_cfg = scenes.get_scene_config(scen)
        env = UAVEnv(scene_cfg, global_cfg=config)
        policy = MLPPolicy(input_dim=10, hidden_dims=[64, 64], output_dim=8)
        trained_policy = train_policy_es(env, policy, config)
        # Save model parameters
        model_path = os.path.join("models", f"M{scen[-1]}.npy")
        np.save(model_path, trained_policy.get_params())
        print(f"Saved trained model to {model_path}")
