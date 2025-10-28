# 修改版：evaluate_transfer 增加了 per-episode metrics 输出与保存
import numpy as np
import matplotlib.pyplot as plt
from env.core import UAVEnv
from env import scenes
from policy.mlp import MLPPolicy
import os

def evaluate_transfer(model_files, scenario_ids, episodes=20, global_config=None, save_metrics=True):
    """
    Evaluate each model (policy) on each scenario and return the success rate matrix.
    Also prints and saves per-model/scenario diagnostics (avg steps, avg kills, escapes).
    """
    n_models = len(model_files)
    n_scen = len(scenario_ids)
    success_matrix = np.zeros((n_models, n_scen))
    detailed_metrics = {}  # store metrics per model/scenario
    # Load models into policies
    policies = []
    for mfile in model_files:
        # Initialize a policy network
        policy = MLPPolicy(input_dim=10, hidden_dims=[64, 64], output_dim=8)
        try:
            params = np.load(mfile, allow_pickle=True)
        except IOError:
            print(f"Model file {mfile} not found. Skipping.")
            policies.append(None)
            continue
        # safety check on params size
        try:
            policy.set_params(params)
        except Exception as e:
            print(f"Error setting params for {mfile}: {e}. Skipping this model.")
            policies.append(None)
            continue
        policies.append(policy)
    # Evaluate each policy on each scenario
    for i, policy in enumerate(policies):
        if policy is None:
            continue
        for j, scen in enumerate(scenario_ids):
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
                total_reward = 0.0
                while not done:
                    action = policy.predict(obs)
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    step_count += 1
                # Gather metrics
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
                'avg_steps': float(np.mean(steps_list)),
                'std_steps': float(np.std(steps_list)),
                'avg_blue_kills': float(np.mean(blue_kills_list)),
                'avg_red_kills': float(np.mean(red_kills_list)),
                'avg_red_escaped': float(np.mean(red_escaped_list)),
                'winners_sample_first5': winners[:5]
            }
            print(f"Model {i+1} on Scene {scen}: win rate {success_rate*100:.1f}%  (avg_steps={np.mean(steps_list):.1f}, avg_blue_kills={np.mean(blue_kills_list):.1f}, avg_red_escaped={np.mean(red_escaped_list):.1f})")
    # Save detailed metrics if desired
    if save_metrics:
        os.makedirs("eval_metrics", exist_ok=True)
        np.save("eval_metrics/transfer_success_matrix.npy", success_matrix)
        import json
        with open("eval_metrics/detailed_metrics.json", "w") as f:
            json.dump(detailed_metrics, f, indent=2)
    # Create heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(success_matrix, vmin=0.0, vmax=1.0, cmap='YlGn')
    # Show text labels
    for i in range(n_models):
        for j in range(n_scen):
            ax.text(j, i, f"{success_matrix[i, j]*100:.0f}%", ha='center', va='center',
                   color=('white' if success_matrix[i,j] < 0.5 else 'black'))
    # Set tick labels
    ax.set_xticks(np.arange(n_scen))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(scenario_ids)
    ax.set_yticklabels([f"M{i+1}" for i in range(n_models)])
    ax.set_xlabel("Test Scenario")
    ax.set_ylabel("Trained Model")
    ax.set_title("Transfer Success Rate")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig("transfer_heatmap.png")
    plt.close(fig)
    return success_matrix, detailed_metrics