import argparse
import os
import yaml
from eval.transfer import evaluate_transfer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate transfer performance of trained models.")
    parser.add_argument("--episodes", type=int, help="Number of episodes for each transfer evaluation (default from config).")
    args = parser.parse_args()
    # Load global config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    if args.episodes is not None:
        eval_eps = args.episodes
    else:
        eval_eps = config.get('eval_episodes', 20)
    # Prepare model files and scenario list
    scenarios = ["S1", "S2", "S3", "S4"]
    model_files = [os.path.join("models", f"M{i}.npy") for i in range(1, 5)]
    # Evaluate transfer matrix
    evaluate_transfer(model_files, scenarios, episodes=eval_eps, global_config=config)
    print("Transfer evaluation complete. Heatmap saved to transfer_heatmap.png.")
