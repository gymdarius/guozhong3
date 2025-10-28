import numpy as np
import matplotlib.pyplot as plt
from env.core import UAVEnv
from env import scenes
from policy.mlp import MLPPolicy

def evaluate_transfer(model_files, scenario_ids, episodes=20, global_config=None):
    """
    Evaluate each model (policy) on each scenario and return the success rate matrix.
    model_files: list of file paths for model parameters (in order corresponding to scenario_ids as training scenarios).
    scenario_ids: list of scenario identifiers (e.g. ['S1','S2','S3','S4']).
    episodes: number of episodes to run for each transfer test.
    global_config: global configuration dict for environment parameters.
    Returns: numpy array of shape (len(model_files), len(scenario_ids)) with win rates.
    Also produces a heatmap saved as 'transfer_heatmap.png'.
    """
    n_models = len(model_files)
    n_scen = len(scenario_ids)
    success_matrix = np.zeros((n_models, n_scen))
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
        policy.set_params(params)
        policies.append(policy)
    # Evaluate each policy on each scenario
    for i, policy in enumerate(policies):
        if policy is None:
            continue
        for j, scen in enumerate(scenario_ids):
            wins = 0
            for ep in range(episodes):
                env = UAVEnv(scenes.get_scene_config(scen), global_cfg=global_config)
                obs = env.reset()
                done = False
                # Use deterministic policy for evaluation
                while not done:
                    action = policy.predict(obs)
                    obs, reward, done, info = env.step(action)
                # Determine winner from info or env
                winner = info.get('winner', None)
                if winner == 'blue':
                    wins += 1
            success_rate = wins / episodes
            success_matrix[i, j] = success_rate
            print(f"Model {i+1} on Scene {scen}: win rate {success_rate*100:.1f}%")
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
    return success_matrix
