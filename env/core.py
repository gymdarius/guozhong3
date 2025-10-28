# 完整文件（基于你已有的 core.py，添加了 escape_penalty 与 terminal_reward 的支持）
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D rendering (if needed)
from mpl_toolkits.mplot3d import art3d   # For adding 2D patches in 3D

from env import scenes

class UAVEnv:
    """
    UAVEnv is a simplified 5v5 UAV confrontation environment.
    It simulates a single-agent (Blue side commander) controlling a formation of 5 UAVs against 5 enemy UAVs (Red side) with scripted behavior.
    The environment handles:
    - State representation (observation of relative positions and various features)
    - Blue actions (movement commands) and their effects
    - Red side scripted behavior per scenario
    - Combat interactions (probabilistic kills, escapes, etc.)
    - Reward calculation and episode termination.
    """
    def __init__(self, scene_cfg, global_cfg=None):
        """
        Initialize environment with a given scenario configuration (scene_cfg).
        global_cfg: optional config dict for global parameters (reward coefficients, max_steps, etc.).
        """
        self.scene = scene_cfg
        self.scenario = scene_cfg['name']
        # Global parameters
        if global_cfg:
            self.alpha = global_cfg.get('alpha', 2.0)
            self.beta = global_cfg.get('beta', 1.5)
            self.gamma = global_cfg.get('gamma', 0.5)
            self.delta = global_cfg.get('delta', 0.1)
            self.max_steps = global_cfg.get('max_steps', 300)
            # 新增：蓝方交战优势参数（0.5 表示无偏，>0.5 表示偏向蓝方）
            self.blue_advantage = global_cfg.get('blue_advantage', 0.7)
            # 新增：每次红方成功逃逸时对蓝方的惩罚（正数，越大蓝方越重视阻止逃逸）
            self.escape_penalty = global_cfg.get('escape_penalty', 8.0)
            # 新增：终局胜利/失败的显著奖励（用于强化学习/ES的终局信号）
            self.terminal_reward = global_cfg.get('terminal_reward', 50.0)
        else:
            # defaults
            self.alpha = 2.0
            self.beta = 1.5
            self.gamma = 0.5
            self.delta = 0.1
            self.max_steps = 300
            self.blue_advantage = 0.7
            self.escape_penalty = 8.0
            self.terminal_reward = 50.0
        # Extract scenario-specific data
        self.initial_blue_positions = np.array(self.scene['blue_init'], dtype=float)
        self.initial_red_positions = np.array(self.scene['red_init'], dtype=float)
        self.blue_types = self.scene.get('blue_types', ['F']*5)
        self.red_types = self.scene.get('red_types', ['F']*5)
        self.weapon_range = self.scene.get('weapon_range', 30.0)
        # Optional scenario parameters
        self.jammer_range = self.scene.get('jammer_range', self.weapon_range)
        self.safe_center = self.scene.get('safe_center', None)
        self.safe_radius = self.scene.get('safe_radius', None)
        self.decoy_threshold = self.scene.get('decoy_threshold', 30.0)
        self.blue_recon_index = self.scene.get('blue_recon_index', None)
        self.red_jammer_indices = self.scene.get('red_jammer_indices', [])
        self.red_decoy_index = self.scene.get('red_decoy_index', None)
        # Altitude levels (discrete: 0,1,2 corresponding to 0,50,100 height for simplicity)
        self.altitude_levels = [0.0, 50.0, 100.0]
        # Internal state
        self.blue_positions = None
        self.red_positions = None
        self.blue_alive = None
        self.red_alive = None
        self.red_escaped = None  # for scenario S3
        self.blue_kills = 0
        self.red_kills = 0
        self.step_count = 0
        # For reward calculation
        self.prev_kill_diff = 0
        self.prev_progress = 0.0
        # track previous escaped count for reward shaping
        self.prev_escaped_count = 0
        # Store initial counts
        self.num_blue_init = len(self.initial_blue_positions)
        self.num_red_init = len(self.initial_red_positions)
    
    def reset(self):
        """Reset the environment to the initial state. Returns the initial observation."""
        # Initialize positions and statuses
        self.blue_positions = np.array(self.initial_blue_positions, dtype=float)
        self.red_positions = np.array(self.initial_red_positions, dtype=float)
        self.blue_alive = np.array([True]*len(self.blue_positions))
        self.red_alive = np.array([True]*len(self.red_positions))
        self.red_escaped = np.array([False]*len(self.red_positions))
        # Reset step and counters
        self.blue_kills = 0
        self.red_kills = 0
        self.step_count = 0
        # Determine initial altitude level for Blue (assume all blue at same level)
        # We'll derive from first Blue UAV position:
        initial_alt = self.blue_positions[0, 2]
        # Find nearest altitude level index (更稳健)
        self.blue_alt_level = int(np.argmin([abs(initial_alt - a) for a in self.altitude_levels]))
        # Add slight random jitter to initial positions for stochasticity
        # (to avoid overfitting to one configuration)
        jitter = np.random.uniform(-5.0, 5.0, size=self.blue_positions.shape)
        self.blue_positions += jitter
        jitter = np.random.uniform(-5.0, 5.0, size=self.red_positions.shape)
        self.red_positions += jitter
        # Compute initial progress for reward tracking
        self.prev_kill_diff = 0
        self.prev_progress = self._compute_goal_progress()
        # Reset prev escaped count
        self.prev_escaped_count = 0
        # Return initial observation
        return self._get_observation()
    
    def step(self, action):
        """
        Apply the Blue agent's action (0-7) and advance the environment by one step.
        Returns: obs, reward, done, info
        """
        # Increase step count
        self.step_count += 1
        # Map action to effect on Blue formation
        self._apply_blue_action(action)
        # Apply enemy scripted behavior according to scenario
        self._apply_enemy_behavior()
        # Check combat interactions (probabilistic kills, escapes)
        self._handle_combat()
        # Compute reward (dense shaping)
        obs = self._get_observation()  # new observation after moves and combat
        reward = self._compute_reward(obs)
        # Check termination conditions
        done, winner = self._check_done()
        # Terminal bonus/penalty
        if done:
            if winner == 'blue':
                reward += self.terminal_reward
            elif winner == 'red':
                reward -= self.terminal_reward
        info = {}
        if done:
            info['winner'] = winner
        return obs, reward, done, info
    
    def _apply_blue_action(self, action):
        """Update Blue positions based on the given action (0-7)."""
        # Compute Blue and Red centers for reference
        blue_alive_positions = self.blue_positions[self.blue_alive]
        red_alive_positions = self.red_positions[self.red_alive]
        if len(blue_alive_positions) == 0 or len(red_alive_positions) == 0:
            return  # no moves if either side has no units
        blue_center = np.mean(blue_alive_positions, axis=0)
        red_center = np.mean(red_alive_positions, axis=0)
        # Direction vector from Blue center to Red center (in horizontal plane)
        vec_to_enemy = red_center - blue_center
        # Only consider horizontal components for direction
        vec2d = np.array([vec_to_enemy[0], vec_to_enemy[1], 0.0])
        # If enemy and blue centers coincide (should not usually happen initially), skip
        if np.linalg.norm(vec2d) < 1e-6:
            direction = np.array([0.0, 0.0, 0.0])
        else:
            direction = vec2d / np.linalg.norm(vec2d)
        # Compute a perpendicular left direction in horizontal plane
        left_dir = np.cross(np.array([0.0, 0.0, 1.0]), direction)
        if np.linalg.norm(left_dir) < 1e-6:
            left_dir = np.array([0.0, 0.0, 0.0])
        else:
            left_dir = left_dir / np.linalg.norm(left_dir)
        move_step = 10.0  # distance to move for directional actions
        # Action effects
        if action == 0:   # Forward (move toward enemy)
            self.blue_positions[self.blue_alive] += direction * move_step
        elif action == 1: # Backward (move away from enemy)
            self.blue_positions[self.blue_alive] -= direction * move_step
        elif action == 2: # Left strafe
            self.blue_positions[self.blue_alive] += left_dir * move_step
        elif action == 3: # Right strafe
            self.blue_positions[self.blue_alive] -= left_dir * move_step
        elif action == 4: # Ascend
            if self.blue_alt_level < 2:
                self.blue_alt_level += 1
                new_alt = self.altitude_levels[self.blue_alt_level]
                # change altitude of all alive Blue
                self.blue_positions[self.blue_alive, 2] = new_alt
        elif action == 5: # Descend
            if self.blue_alt_level > 0:
                self.blue_alt_level -= 1
                new_alt = self.altitude_levels[self.blue_alt_level]
                self.blue_positions[self.blue_alive, 2] = new_alt
        elif action == 6: # Gather (tighten formation)
            center = np.mean(self.blue_positions[self.blue_alive], axis=0)
            factor = 0.8
            for i, alive in enumerate(self.blue_alive):
                if alive:
                    self.blue_positions[i] = center + (self.blue_positions[i] - center) * factor
        elif action == 7: # Disperse (spread out formation)
            center = np.mean(self.blue_positions[self.blue_alive], axis=0)
            factor = 1.2
            for i, alive in enumerate(self.blue_alive):
                if alive:
                    self.blue_positions[i] = center + (self.blue_positions[i] - center) * factor
        else:
            # If an invalid action is given, do nothing
            pass
    
    def _apply_enemy_behavior(self):
        """Update Red positions according to scenario's scripted logic."""
        if self.scenario == 'S1':
            # Red moves straightforward toward Blue
            blue_alive_positions = self.blue_positions[self.blue_alive]
            if len(blue_alive_positions) == 0:
                return
            blue_center = np.mean(blue_alive_positions, axis=0)
            red_alive_indices = np.where(self.red_alive)[0]
            for i in red_alive_indices:
                # Move each alive Red toward Blue center
                direction = blue_center - self.red_positions[i]
                direction[2] = 0.0  # ignore altitude difference in movement
                dist = np.linalg.norm(direction)
                if dist > 1e-6:
                    step_dir = direction / dist
                    self.red_positions[i] += step_dir * 10.0
        elif self.scenario == 'S2':
            # Jammers move toward Blue recon (if alive), fighters move toward Blue center
            blue_alive_positions = self.blue_positions[self.blue_alive]
            if len(blue_alive_positions) == 0:
                return
            blue_center = np.mean(blue_alive_positions, axis=0)
            recon_alive = False
            recon_pos = None
            if self.blue_recon_index is not None and self.blue_recon_index < len(self.blue_alive) and self.blue_alive[self.blue_recon_index]:
                recon_alive = True
                recon_pos = self.blue_positions[self.blue_recon_index]
            red_alive_indices = np.where(self.red_alive)[0]
            for i in red_alive_indices:
                if self.red_types[i] == 'J':
                    # Jammer
                    target = recon_pos if (recon_alive and recon_pos is not None) else blue_center
                    direction = target - self.red_positions[i]
                else:
                    # Fighter
                    direction = blue_center - self.red_positions[i]
                direction[2] = 0.0
                dist = np.linalg.norm(direction)
                if dist > 1e-6:
                    step_dir = direction / dist
                    self.red_positions[i] += step_dir * 10.0
        elif self.scenario == 'S3':
            # Red tries to escape outward from safe center
            if self.safe_center is None or self.safe_radius is None:
                return
            red_alive_indices = np.where(self.red_alive)[0]
            for i in red_alive_indices:
                # If not already escaped, move away from safe center
                if not self.red_escaped[i]:
                    direction = self.red_positions[i] - self.safe_center
                    direction[2] = 0.0
                    dist = np.linalg.norm(direction)
                    if dist < 1e-6:
                        # if at center exactly, move in some fixed direction
                        direction = np.array([1.0, 0.0, 0.0])
                        dist = 1.0
                    step_dir = direction / dist
                    self.red_positions[i] += step_dir * 10.0
                    # Check if escapes this step
                    new_dist = np.linalg.norm(self.red_positions[i] - self.safe_center)
                    if new_dist >= self.safe_radius:
                        # Mark as escaped and remove from battle
                        self.red_escaped[i] = True
                        self.red_alive[i] = False
        elif self.scenario == 'S4':
            # Decoy and real Red behavior
            blue_alive_positions = self.blue_positions[self.blue_alive]
            if len(blue_alive_positions) == 0:
                return
            blue_center = np.mean(blue_alive_positions, axis=0)
            red_alive_indices = np.where(self.red_alive)[0]
            decoy_idx = self.red_decoy_index
            for i in red_alive_indices:
                if i == decoy_idx:
                    # Decoy logic
                    # If Blue gets too close, decoy moves away from Blue to lure them further
                    dist = np.linalg.norm(self.red_positions[i] - blue_center)
                    if dist < self.decoy_threshold:
                        # move further from Blue
                        direction = self.red_positions[i] - blue_center
                        if np.linalg.norm(direction) < 1e-6:
                            continue
                        step_dir = direction / np.linalg.norm(direction)
                        self.red_positions[i] += step_dir * 10.0
                    else:
                        # Otherwise, minimal movement or hold position (could also move slowly forward/back)
                        # Here do nothing (hold position)
                        pass
                else:
                    # Real Red units move toward Blue (attack)
                    direction = blue_center - self.red_positions[i]
                    direction[2] = 0.0
                    dist = np.linalg.norm(direction)
                    if dist > 1e-6:
                        step_dir = direction / dist
                        self.red_positions[i] += step_dir * 10.0
        else:
            # Default: no special movement
            return
    
    def _handle_combat(self):
        """Handle weapon range combat using probabilistic resolution:
        For pairs within weapon_range, there is a distance-dependent chance that a kill occurs.
        If a kill occurs, the winner is decided probabilistically biased toward Blue (configurable).
        This prevents guaranteed mutual-destruction and lets the agent exploit blue advantage.
        """
        blue_alive_indices = np.where(self.blue_alive)[0]
        red_alive_indices = np.where(self.red_alive)[0]
        if len(blue_alive_indices) == 0 or len(red_alive_indices) == 0:
            return
        # Positions of alive units
        blue_positions_alive = self.blue_positions[blue_alive_indices]
        red_positions_alive = self.red_positions[red_alive_indices]
        # Compute pairwise distances
        diff = blue_positions_alive[:, np.newaxis, :] - red_positions_alive[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)  # shape (nb, nr)
        # Flatten pair list and sort by distance ascending so close encounters resolve first
        pairs = []
        for ib in range(dist_matrix.shape[0]):
            for ir in range(dist_matrix.shape[1]):
                d = dist_matrix[ib, ir]
                pairs.append((d, blue_alive_indices[ib], red_alive_indices[ir]))
        pairs.sort(key=lambda x: x[0])
        weapon_range = float(self.weapon_range)
        # Parameters for probabilistic kill
        max_kill_prob_scale = 0.9  # maximum total probability that a kill happens at zero distance
        blue_favor = float(self.blue_advantage)  # e.g., 0.7 => 70% of kills favor blue when kill occurs
        for d, b_idx, r_idx in pairs:
            # Skip if either already dead due to earlier resolution
            if not (self.blue_alive[b_idx] and self.red_alive[r_idx]):
                continue
            if d >= weapon_range:
                continue
            # compute base proximity factor (0..1)
            proximity = max(0.0, 1.0 - (d / weapon_range))
            # total probability that a kill occurs for this encounter
            kill_total_prob = max_kill_prob_scale * proximity
            if np.random.rand() < kill_total_prob:
                # a kill occurs; decide who dies (favor blue)
                if np.random.rand() < blue_favor:
                    # Blue scores the kill (red dies)
                    self.red_alive[r_idx] = False
                    self.blue_kills += 1
                else:
                    # Red scores the kill (blue dies)
                    self.blue_alive[b_idx] = False
                    self.red_kills += 1
            # else: no kill this pair this timestep
        # done
    
    def _get_observation(self):
        """Compute the observation vector based on current state."""
        # relative_pos (3,) Blue centroid relative to Red centroid
        # If either side has no alive, define relative_pos as zeros
        if np.any(self.blue_alive) and np.any(self.red_alive):
            blue_center = np.mean(self.blue_positions[self.blue_alive], axis=0)
            red_center = np.mean(self.red_positions[self.red_alive], axis=0)
            relative_pos = blue_center - red_center
        else:
            relative_pos = np.array([0.0, 0.0, 0.0])
        # own_alive_ratio and enemy_alive_ratio
        own_alive_ratio = np.sum(self.blue_alive) / len(self.blue_alive)
        enemy_alive_ratio = np.sum(self.red_alive) / len(self.red_alive)
        # avg_height_level (0~2)
        avg_height_level = float(self.blue_alt_level)
        # nearest_enemy_dist
        if np.any(self.blue_alive) and np.any(self.red_alive):
            # compute nearest distance between any alive blue-red
            diff = self.blue_positions[self.blue_alive][:, np.newaxis, :] - self.red_positions[self.red_alive][np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            nearest_enemy_dist = float(np.min(dists))
        else:
            nearest_enemy_dist = 0.0
        # interference_level (0~1)
        interference_level = 0.0
        if self.scenario == 'S2':
            # if any jammer alive and Blue recon alive
            if self.blue_recon_index is not None and self.blue_recon_index < len(self.blue_alive) and self.blue_alive[self.blue_recon_index]:
                target_pos = self.blue_positions[self.blue_recon_index]
            elif np.any(self.blue_alive):
                target_pos = np.mean(self.blue_positions[self.blue_alive], axis=0)
            else:
                target_pos = None
            count_in_range = 0
            total_jammers = 0
            if target_pos is not None:
                for j_idx in self.red_jammer_indices:
                    if j_idx < len(self.red_alive) and self.red_alive[j_idx]:
                        total_jammers += 1
                        dist = np.linalg.norm(self.red_positions[j_idx] - target_pos)
                        if dist < self.jammer_range:
                            count_in_range += 1
            if total_jammers > 0:
                interference_level = count_in_range / total_jammers
        # bait_confidence (0~1)
        bait_confidence = 0.0
        if self.scenario == 'S4':
            if self.red_decoy_index is not None and self.red_decoy_index < len(self.red_alive) and self.red_alive[self.red_decoy_index]:
                # distance between Blue center and decoy
                if np.any(self.blue_alive):
                    blue_center = np.mean(self.blue_positions[self.blue_alive], axis=0)
                    decoy_pos = self.red_positions[self.red_decoy_index]
                    dist = np.linalg.norm(decoy_pos - blue_center)
                    if dist < self.decoy_threshold:
                        bait_confidence = 1.0
                    else:
                        bait_confidence = 0.0
            else:
                # decoy dead -> set to 0 (no bait present)
                bait_confidence = 0.0
        # goal_progress (0~1)
        goal_progress = self._compute_goal_progress()
        # Construct observation array
        obs = np.concatenate([
            relative_pos.astype(float),
            [own_alive_ratio, enemy_alive_ratio, avg_height_level, nearest_enemy_dist, interference_level, bait_confidence, goal_progress]
        ]).astype(float)
        return obs
    
    def _compute_goal_progress(self):
        """Compute the current goal progress metric (scenario-dependent, 0 to 1)."""
        progress = 0.0
        if self.scenario == 'S1':
            # fraction of enemies eliminated
            initial_red = self.num_red_init
            eliminated = self.blue_kills  # number of Red killed by Blue
            progress = eliminated / initial_red
        elif self.scenario == 'S2':
            # combine fraction of enemies killed and recon survival
            initial_red = self.num_red_init
            kill_frac = self.blue_kills / initial_red
            recon_alive = 1.0 if (self.blue_recon_index is not None and self.blue_recon_index < self.num_blue_init and self.blue_alive[self.blue_recon_index]) else 0.0
            progress = 0.5 * kill_frac + 0.5 * recon_alive
        elif self.scenario == 'S3':
            # 1 - (escaped_count/3), capped between 0 and 1
            esc = np.sum(self.red_escaped)
            progress = 1.0 - min(1.0, esc / 3.0)
        elif self.scenario == 'S4':
            # fraction of real red killed minus penalty if decoy killed prematurely
            # Determine real Red count and killed
            real_indices = [i for i, t in enumerate(self.red_types) if t != 'D']
            real_total = len(real_indices)
            real_alive = sum(1 for i in real_indices if i < len(self.red_alive) and self.red_alive[i])
            real_killed_count = real_total - real_alive
            progress = real_killed_count / real_total if real_total > 0 else 0.0
            # Penalty if decoy is already killed while any real still alive
            if self.red_decoy_index is not None:
                if self.red_decoy_index < len(self.red_alive) and not self.red_alive[self.red_decoy_index] and real_alive > 0:
                    progress -= 0.5
                    if progress < 0:
                        progress = 0.0
        else:
            progress = 0.0
        return progress
    
    def _compute_reward(self, obs):
        """Compute reward for the current step based on changes in key metrics."""
        # Kill difference increment
        current_kill_diff = self.blue_kills - self.red_kills
        kill_diff_increment = current_kill_diff - self.prev_kill_diff
        self.prev_kill_diff = current_kill_diff
        # Objective progress change
        current_progress = obs[-1]  # goal_progress is last element of obs
        progress_change = current_progress - self.prev_progress
        self.prev_progress = current_progress
        # Survival difference (Blue vs Red)
        blue_alive_ratio = obs[3]  # own_alive_ratio at index 3 if relative_pos is 0-2
        red_alive_ratio = obs[4]   # enemy_alive_ratio at index 4
        survival_diff = blue_alive_ratio - red_alive_ratio
        # Escape increment penalty (特别针对 S3)
        current_escaped = np.sum(self.red_escaped)
        escape_increment = int(current_escaped - self.prev_escaped_count)
        # Update prev escaped count for next step
        self.prev_escaped_count = current_escaped
        escape_penalty_term = - self.escape_penalty * escape_increment if escape_increment > 0 else 0.0
        # Time penalty (for each step)
        time_penalty = 1.0
        # Total reward
        reward = (self.alpha * kill_diff_increment) + (self.beta * progress_change) + (self.gamma * survival_diff) + escape_penalty_term - (self.delta * time_penalty)
        return reward
    
    def _check_done(self):
        """Check if the episode is finished and determine the winner."""
        winner = None
        # If maximum steps reached, decide winner by scenario-specific conditions
        if self.step_count >= self.max_steps:
            if self.scenario == 'S1':
                # Compare kill counts
                if self.blue_kills > self.red_kills:
                    winner = 'blue'
                elif self.red_kills > self.blue_kills:
                    winner = 'red'
                else:
                    winner = 'draw'
            elif self.scenario == 'S2':
                # Use score = Blue_kills + (recon_alive ?1:0) vs Red_kills
                blue_score = self.blue_kills + (1 if (self.blue_recon_index is not None and self.blue_recon_index < len(self.blue_alive) and self.blue_alive[self.blue_recon_index]) else 0)
                red_score = self.red_kills
                if blue_score > red_score:
                    winner = 'blue'
                elif red_score > blue_score:
                    winner = 'red'
                else:
                    winner = 'draw'
            elif self.scenario == 'S3':
                # If Red escapes >=3 then Red wins, else Blue wins (if time up without breakout)
                if np.sum(self.red_escaped) >= 3:
                    winner = 'red'
                else:
                    winner = 'blue'
            elif self.scenario == 'S4':
                # If any real Red alive, Red wins (Blue failed to eliminate all threats in time); else Blue wins
                real_indices = [i for i, t in enumerate(self.red_types) if t != 'D']
                any_real_alive = any(i < len(self.red_alive) and self.red_alive[i] for i in real_indices)
                if any_real_alive:
                    winner = 'red'
                else:
                    winner = 'blue'
            else:
                winner = 'draw'
            return True, winner
        # Else, check if one side is completely defeated or scenario conditions met
        blue_alive_count = np.sum(self.blue_alive)
        red_alive_count = np.sum(self.red_alive)
        if blue_alive_count == 0:
            winner = 'red'
            return True, winner
        if red_alive_count == 0:
            if self.scenario == 'S3':
                # All Red in battle eliminated, check escapes
                if np.sum(self.red_escaped) < 3:
                    winner = 'blue'
                else:
                    winner = 'red'
            else:
                winner = 'blue'
            return True, winner
        if self.scenario == 'S3':
            # If Red breakout success condition
            if np.sum(self.red_escaped) >= 3:
                winner = 'red'
                return True, winner
        if self.scenario == 'S4':
            # If all real red destroyed
            real_indices = [i for i, t in enumerate(self.red_types) if t != 'D']
            all_reals_dead = all((i < len(self.red_alive) and not self.red_alive[i]) for i in real_indices)
            if all_reals_dead:
                winner = 'blue'
                return True, winner
        # Otherwise not done
        return False, winner
    
    def render(self, mode='3d'):
        """Render the current state in a 3D scatter plot (for quick visualization)."""
        if mode != '3d':
            print("Only 3D mode is supported for rendering.")
            return
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plot Blue alive units (blue)
        blue_alive = np.where(self.blue_alive)[0]
        red_alive = np.where(self.red_alive)[0]
        if len(blue_alive) > 0:
            ax.scatter(self.blue_positions[blue_alive,0], self.blue_positions[blue_alive,1], self.blue_positions[blue_alive,2], c='b', label='Blue')
        if len(red_alive) > 0:
            # Decoy vs real distinction
            if self.scenario == 'S4' and self.red_decoy_index is not None and self.red_decoy_index in red_alive:
                # Plot decoy separately as yellow star
                dec_idx = self.red_decoy_index
                red_alive = red_alive.tolist()
                if dec_idx in red_alive:
                    red_alive.remove(dec_idx)
                    ax.scatter(self.red_positions[dec_idx,0], self.red_positions[dec_idx,1], self.red_positions[dec_idx,2], c='y', marker='*', label='Red Decoy')
                if len(red_alive) > 0:
                    ax.scatter(self.red_positions[red_alive,0], self.red_positions[red_alive,1], self.red_positions[red_alive,2], c='r', label='Red')
            else:
                ax.scatter(self.red_positions[red_alive,0], self.red_positions[red_alive,1], self.red_positions[red_alive,2], c='r', label='Red')
        ax.set_xlim([-100, 120])
        ax.set_ylim([-100, 120])
        ax.set_zlim([0, 120])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()