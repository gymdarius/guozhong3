import argparse
import os
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt

from env import scenes
from env.core import UAVEnv
from policy.mlp import MLPPolicy
from policy.mlp_continuous import MLPPolicyContinuous

"""
替换/增强版 run_eval.py

说明：
- 原来的 evaluate_transfer 假定所有模型均为离散 MLPPolicy，导致当模型是连续策略（MLPPolicyContinuous）
  时因参数尺寸不匹配而跳过。为兼容当前仓库中可能混合的模型类型（离散/连续），这里实现了一个
  独立的评估逻辑：对每个模型文件先尝试把参数加载到连续策略类 MLPPolicyContinuous（优先），
  如果不匹配再尝试离散策略 MLPPolicy；两者都不行则跳过该模型并打印原因。
- 对每个(模型, 场景)按 `episodes` 次运行并统计胜率与若干度量，最终保存 heatmap 与详细的 metrics json。
- 使用方法（与原脚本兼容）：
    python run_eval.py --episodes 20
"""

def evaluate_models(model_files, scenario_ids, episodes=20, global_config=None, save_metrics=True):
    n_models = len(model_files)
    n_scen = len(scenario_ids)
    success_matrix = np.zeros((n_models, n_scen))
    detailed_metrics = {}

    # Load and build policy objects for each model file
    policies = []
    policy_types = []  # "continuous", "discrete", or None
    for mfile in model_files:
        if not os.path.exists(mfile):
            print(f"Model file {mfile} not found. Skipping.")
            policies.append(None)
            policy_types.append(None)
            continue
        # load params
        try:
            params = np.load(mfile, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {mfile}: {e}. Skipping.")
            policies.append(None)
            policy_types.append(None)
            continue
        # try continuous policy first
        try:
            policy_c = MLPPolicyContinuous(input_dim=10, hidden_dims=[64,64], action_dim=4)
            policy_c.set_params(params)
            policies.append(policy_c)
            policy_types.append("continuous")
            print(f"Loaded continuous policy from {mfile}")
            continue
        except Exception:
            # try discrete policy
            try:
                policy_d = MLPPolicy(input_dim=10, hidden_dims=[64,64], output_dim=8)
                policy_d.set_params(params)
                policies.append(policy_d)
                policy_types.append("discrete")
                print(f"Loaded discrete policy from {mfile}")
                continue
            except Exception as e2:
                print(f"Error setting params for {mfile}: {e2}. Skipping this model.")
                policies.append(None)
                policy_types.append(None)
                continue

    # Evaluate each policy on each scenario
    for i, policy in enumerate(policies):
        for j, scen in enumerate(scenario_ids):
            if policy is None:
                # leave success_matrix[i,j] as 0
                detailed_metrics[f"M{i+1}_{scen}"] = {
                    'win_rate': None,
                    'episodes': episodes,
                    'wins': 0,
                    'avg_steps': None,
                    'std_steps': None,
                    'avg_blue_kills': None,
                    'avg_red_kills': None,
                    'avg_red_escaped': None,
                    'winners_sample_first5': []
                }
                continue

            wins = 0
            steps_list = []
            blue_kills_list = []
            red_kills_list = []
            red_escaped_list = []
            winners = []

            for ep in range(episodes):
                env = UAVEnv(scenes.get_scene_config(scen), global_cfg=global_config)
                obs = env.reset()
                done = False
                step_count = 0
                # use greedy predict for evaluation (deterministic)
                while not done:
                    try:
                        action = policy.predict(obs)
                    except Exception:
                        # fallback to sampling if predict not available
                        action = policy.sample_action(obs)
                    obs, reward, done, info = env.step(action)
                    step_count += 1
                winner = info.get('winner', None)
                winners.append(winner)
                if winner == 'blue':
                    wins += 1
                steps_list.append(step_count)
                blue_kills_list.append(env.blue_kills)
                red_kills_list.append(env.red_kills)
                red_escaped_list.append(int(np.sum(env.red_escaped)))

            success_rate = wins / episodes
            success_matrix[i, j] = success_rate
            detailed_metrics[f"M{i+1}_{scen}"] = {
                'win_rate': success_rate,
                'episodes': episodes,
                'wins': wins,
                'avg_steps': float(np.mean(steps_list)) if steps_list else None,
                'std_steps': float(np.std(steps_list)) if steps_list else None,
                'avg_blue_kills': float(np.mean(blue_kills_list)) if blue_kills_list else None,
                'avg_red_kills': float(np.mean(red_kills_list)) if red_kills_list else None,
                'avg_red_escaped': float(np.mean(red_escaped_list)) if red_escaped_list else None,
                'winners_sample_first5': winners[:5]
            }
            print(f"Model {i+1} ({policy_types[i]}) on Scene {scen}: win rate {success_rate*100:.1f}%  (avg_steps={np.mean(steps_list):.1f}, avg_blue_kills={np.mean(blue_kills_list):.1f}, avg_red_escaped={np.mean(red_escaped_list):.1f})")

    # Save metrics and heatmap
    if save_metrics:
        os.makedirs("eval_metrics", exist_ok=True)
        np.save("eval_metrics/transfer_success_matrix.npy", success_matrix)
        with open("eval_metrics/detailed_metrics.json", "w") as f:
            json.dump(detailed_metrics, f, indent=2)
    # Plot heatmap (handle case with zero models gracefully)
    if success_matrix.size > 0 and success_matrix.shape[0] > 0:
        fig, ax = plt.subplots()
        im = ax.imshow(success_matrix, vmin=0.0, vmax=1.0, cmap='YlGn')
        for ii in range(success_matrix.shape[0]):
            for jj in range(success_matrix.shape[1]):
                ax.text(jj, ii, f"{success_matrix[ii, jj]*100:.0f}%", ha='center', va='center',
                        color=('white' if success_matrix[ii,jj] < 0.5 else 'black'))
        ax.set_xticks(np.arange(len(scenario_ids)))
        ax.set_yticks(np.arange(len(model_files)))
        ax.set_xticklabels(scenario_ids)
        ax.set_yticklabels([os.path.basename(m) for m in model_files])
        ax.set_xlabel("Test Scenario")
        ax.set_ylabel("Trained Model")
        ax.set_title("Transfer Success Rate")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        plt.savefig("transfer_heatmap.png")
        plt.close(fig)
        print("Saved transfer_heatmap.png")
    else:
        print("No valid models found to plot heatmap.")

    return success_matrix, detailed_metrics

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
    # Prepare model files and scenario list (根据你当前模型命名)
    scenarios = ["S1", "S2", "S3", "S4"]
    model_files = [os.path.join("models", f"M_S{i}_cont.npy") for i in range(1, 5)]
    # Evaluate transfer matrix using the robust loader/evaluator above
    evaluate_models(model_files, scenarios, episodes=eval_eps, global_config=config, save_metrics=True)
    print("Transfer evaluation complete.")