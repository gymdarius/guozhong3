import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import art3d

from env.core import UAVEnv
from env import scenes
from policy.mlp import MLPPolicy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a scenario with a trained model or random policy.")
    parser.add_argument("--scenario", type=str, default="S1", help="Scenario to visualize (S1, S2, S3, S4 or ALL).")
    parser.add_argument("--model", type=str, default=None, help="Path to model .npy file (if not provided, use corresponding models/M#.npy or random policy).")
    args = parser.parse_args()
    scenarios = []
    if args.scenario.upper() == "ALL":
        scenarios = ["S1", "S2", "S3", "S4"]
    else:
        scenarios = [args.scenario.upper()]
    os.makedirs("videos", exist_ok=True)
    for scen in scenarios:
        # Determine model file
        model_file = args.model
        if model_file is None:
            # use default models/M#.npy
            scen_num = scen[1:] if scen.startswith('S') else scen
            model_file = os.path.join("models", f"M{scen_num}.npy")
        # Load model if exists
        policy = MLPPolicy(input_dim=10, hidden_dims=[64, 64], output_dim=8)
        if os.path.exists(model_file):
            params = np.load(model_file, allow_pickle=True)
            policy.set_params(params)
            print(f"Loaded model parameters from {model_file}")
        else:
            print(f"Model file {model_file} not found. Using untrained policy (random behavior).")
        # Create environment for scenario
        env = UAVEnv(scenes.get_scene_config(scen))
        obs = env.reset()
        frames = []
        total_reward = 0.0
        done = False
        step_num = 0
        # Simulate one episode
        while not done:
            # Choose action (greedy from policy)
            action = policy.predict(obs)
            # We can also randomize a bit if desired: but deterministic for visualization
            obs, reward, done, info = env.step(action)
            total_reward += reward
            # Record positions for this step
            step_num += 1
            blue_positions = env.blue_positions[env.blue_alive].copy()
            red_positions = env.red_positions[env.red_alive].copy()
            # Identify decoy and safe zone if applicable
            decoy_pos = None
            if scen == "S4" and env.red_decoy_index is not None:
                if env.red_decoy_index < len(env.red_alive) and not env.red_alive[env.red_decoy_index]:
                    # decoy dead; just treat as normal in rendering now
                    decoy_pos = None
                elif env.red_decoy_index < len(env.red_alive):
                    decoy_pos = env.red_positions[env.red_decoy_index].copy()
            safe_radius = None
            if scen == "S3" and env.safe_center is not None:
                safe_radius = env.safe_radius
            frames.append({
                'blue': blue_positions,
                'red': red_positions,
                'decoy': decoy_pos,
                'blue_kills': env.blue_kills,
                'red_kills': env.red_kills,
                'step': step_num,
                'total_reward': total_reward,
                'safe_center': env.safe_center.copy() if env.safe_center is not None else None,
                'safe_radius': safe_radius
            })
        # Set up 3D plot for animation
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plot styling: Blue as blue circles, Red as red triangles, Decoy as yellow star, safe zone as green circle
        blue_scatter = ax.scatter([], [], [], c='b', marker='o', label='Blue')
        red_scatter = ax.scatter([], [], [], c='r', marker='^', label='Red')
        decoy_scatter = None
        if any(frame['decoy'] is not None for frame in frames):
            decoy_scatter = ax.scatter([], [], [], c='y', marker='*', label='Decoy')
        # Draw safe zone if present
        safe_circle = None
        if frames[0]['safe_center'] is not None and frames[0]['safe_radius'] is not None:
            sx, sy, _ = frames[0]['safe_center']
            safe_radius = frames[0]['safe_radius']
            circle = plt.Circle((sx, sy), safe_radius, color='g', alpha=0.2)
            ax.add_patch(circle)
            art3d.pathpatch_2d_to_3d(circle, z=0, zdir="z")
            safe_circle = circle
            ax.plot([], [], [], color='g', label='Safe Zone')
        # Add legend and axis limits
        ax.set_xlim(-100, 120)
        ax.set_ylim(-100, 120)
        ax.set_zlim(0, 120)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='upper right')
        # HUD text
        hud_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
        def update(frame):
            # frame is a dict from frames list
            blue = frame['blue']
            red = frame['red']
            # Update scatter data
            if len(blue) > 0:
                blue_scatter._offsets3d = (blue[:,0], blue[:,1], blue[:,2])
            else:
                blue_scatter._offsets3d = ([], [], [])
            if len(red) > 0:
                red_scatter._offsets3d = (red[:,0], red[:,1], red[:,2])
            else:
                red_scatter._offsets3d = ([], [], [])
            if decoy_scatter is not None:
                if frame['decoy'] is not None:
                    decoy_scatter._offsets3d = (np.array([frame['decoy'][0]]), 
                                                np.array([frame['decoy'][1]]),
                                                np.array([frame['decoy'][2]]))
                else:
                    decoy_scatter._offsets3d = ([], [], [])
            # Update HUD text
            hud_text.set_text(f"Step: {frame['step']}   Blue kills: {frame['blue_kills']}   Red kills: {frame['red_kills']}   Reward: {frame['total_reward']:.1f}")
            return blue_scatter, red_scatter, decoy_scatter, hud_text
        # Create animation
        ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
        # Save to file
        output_path = os.path.join("videos", f"demo_{scen}.mp4")
        try:
            ani.save(output_path, writer=FFMpegWriter(fps=10))
            print(f"Saved animation to {output_path}")
        except Exception as e:
            print(f"Could not save video (FFMpeg might be missing): {e}")
        plt.close(fig)
