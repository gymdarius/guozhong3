# env/core.py (重写为基于 platform 对象的版本，兼容原来数组接口)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d

from env import scenes
from flat_models.platforms import Fighter, Recon, Jammer, Decoy

class UAVEnv:
    """
    基于 platform 对象的 UAV 对抗环境（单智能体控制整个编队）。
    支持动作：
      - int (0..7) 旧离散动作（向后兼容）
      - np.array length>=3 or length==4: [dx,dy,dz,(formation_scale)]
    单步位置更新采用真三维 kinematics：pos += vel * dt
    """
    def __init__(self, scene_cfg, global_cfg=None):
        self.scene = scene_cfg
        self.scenario = scene_cfg['name']
        # 全局参数
        if global_cfg:
            self.alpha = global_cfg.get('alpha', 2.0)
            self.beta = global_cfg.get('beta', 1.5)
            self.gamma = global_cfg.get('gamma', 0.5)
            self.delta = global_cfg.get('delta', 0.1)
            self.max_steps = global_cfg.get('max_steps', 300)
            self.blue_advantage = global_cfg.get('blue_advantage', 0.7)
            self.escape_penalty = global_cfg.get('escape_penalty', 20.0)
            self.terminal_reward = global_cfg.get('terminal_reward', 50.0)
            self.dt = global_cfg.get('dt', 1.0)
            self.blue_max_speed = global_cfg.get('blue_max_speed', 15.0)
            self.red_max_speed = global_cfg.get('red_max_speed', 10.0)
        else:
            self.alpha = 2.0
            self.beta = 1.5
            self.gamma = 0.5
            self.delta = 0.1
            self.max_steps = 300
            self.blue_advantage = 0.7
            self.escape_penalty = 8.0
            self.terminal_reward = 50.0
            self.dt = 1.0
            self.blue_max_speed = 15.0
            self.red_max_speed = 10.0

        # 场景参数
        self.initial_blue_positions = np.array(self.scene['blue_init'], dtype=float)
        self.initial_red_positions = np.array(self.scene['red_init'], dtype=float)
        self.blue_types = self.scene.get('blue_types', ['F'] * len(self.initial_blue_positions))
        self.red_types = self.scene.get('red_types', ['F'] * len(self.initial_red_positions))
        self.weapon_range = self.scene.get('weapon_range', 30.0)
        self.jammer_range = self.scene.get('jammer_range', self.weapon_range)
        self.safe_center = self.scene.get('safe_center', None)
        self.safe_radius = self.scene.get('safe_radius', None)
        self.decoy_threshold = self.scene.get('decoy_threshold', 30.0)
        self.blue_recon_index = self.scene.get('blue_recon_index', None)
        self.red_jammer_indices = self.scene.get('red_jammer_indices', [])
        self.red_decoy_index = self.scene.get('red_decoy_index', None)

        # 内部状态（两套表示：对象列表 + 数组）
        self.blue_units = []   # list of platform objects
        self.red_units = []
        self.blue_positions = None  # numpy array for external compat
        self.red_positions = None
        self.blue_velocities = None
        self.red_velocities = None
        self.blue_alive = None
        self.red_alive = None
        self.red_escaped = None

        # 计数/奖励追踪
        self.blue_kills = 0
        self.red_kills = 0
        self.step_count = 0
        self.prev_kill_diff = 0
        self.prev_progress = 0.0
        self.prev_escaped_count = 0
        self.num_blue_init = len(self.initial_blue_positions)
        self.num_red_init = len(self.initial_red_positions)

    def _create_unit(self, utype, pos, dt):
        """工厂函数：根据类型字符串创建 platform 对象，返回对象"""
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        # default velocity and angles
        V = 10.0
        Pitch = 0.0
        Heading = 0.0
        kwargs = {'dt': dt}
        if utype == 'F':
            return Fighter([x, y, z], V, Pitch, Heading, dt=dt)
        elif utype == 'R':
            return Recon([x, y, z], V, Pitch, Heading, dt=dt)
        elif utype == 'J':
            return Jammer([x, y, z], V, Pitch, Heading, dt=dt)
        elif utype == 'D':
            return Decoy([x, y, z], V, Pitch, Heading, dt=dt)
        else:
            # fallback to Fighter
            return Fighter([x, y, z], V, Pitch, Heading, dt=dt)

    def reset(self):
        """重置：创建 platform 对象并同步位置数组"""
        self.blue_units = []
        self.red_units = []
        self.blue_alive = np.array([True] * len(self.initial_blue_positions))
        self.red_alive = np.array([True] * len(self.initial_red_positions))
        self.red_escaped = np.array([False] * len(self.initial_red_positions))
        # 创建对象
        for i, pos in enumerate(self.initial_blue_positions):
            u = self._create_unit(self.blue_types[i] if i < len(self.blue_types) else 'F', pos, self.dt)
            u.X, u.Y, u.Z = float(pos[0]), float(pos[1]), float(pos[2])
            u.vel = np.zeros(3, dtype=float)
            self.blue_units.append(u)
        for i, pos in enumerate(self.initial_red_positions):
            u = self._create_unit(self.red_types[i] if i < len(self.red_types) else 'F', pos, self.dt)
            u.X, u.Y, u.Z = float(pos[0]), float(pos[1]), float(pos[2])
            u.vel = np.zeros(3, dtype=float)
            self.red_units.append(u)
        # 同步位置数组与速度数组
        self._sync_arrays_from_units()
        # counters
        self.blue_kills = 0
        self.red_kills = 0
        self.step_count = 0
        self.prev_kill_diff = 0
        self.prev_progress = self._compute_goal_progress()
        self.prev_escaped_count = 0
        return self._get_observation()

    def _sync_arrays_from_units(self):
        """把 platform 对象的 X,Y,Z 与 vel 同步回数组（供 render / eval / 旧代码使用）"""
        self.blue_positions = np.array([[u.X, u.Y, u.Z] for u in self.blue_units], dtype=float)
        self.red_positions = np.array([[u.X, u.Y, u.Z] for u in self.red_units], dtype=float)
        self.blue_velocities = np.array([getattr(u, 'vel', np.zeros(3)) for u in self.blue_units], dtype=float)
        self.red_velocities = np.array([getattr(u, 'vel', np.zeros(3)) for u in self.red_units], dtype=float)

    def _sync_units_from_arrays(self):
        """如果外部直接改了 positions/velocities，可以调用此函数写回到对象（目前未必要）"""
        for i, u in enumerate(self.blue_units):
            u.X, u.Y, u.Z = self.blue_positions[i]
            u.vel = self.blue_velocities[i]
        for i, u in enumerate(self.red_units):
            u.X, u.Y, u.Z = self.red_positions[i]
            u.vel = self.red_velocities[i]

    def step(self, action):
        """一步：应用动作 -> 敌方脚本 -> 更新位置（pos += vel*dt）-> 处理交战 -> 返回 obs/reward/done/info"""
        self.step_count += 1
        # apply blue action (supports discrete ints and continuous vectors)
        self._apply_blue_action(action)
        # apply scripted enemy moves (也会设置 red_units[].vel)
        self._apply_enemy_behavior()
        # integrate positions for all alive units
        for i, alive in enumerate(self.blue_alive):
            if alive:
                v = getattr(self.blue_units[i], 'vel', np.zeros(3))
                self.blue_units[i].X += v[0] * self.dt
                self.blue_units[i].Y += v[1] * self.dt
                self.blue_units[i].Z += v[2] * self.dt
        for i, alive in enumerate(self.red_alive):
            if alive:
                v = getattr(self.red_units[i], 'vel', np.zeros(3))
                self.red_units[i].X += v[0] * self.dt
                self.red_units[i].Y += v[1] * self.dt
                self.red_units[i].Z += v[2] * self.dt
        # S3: 移动完成后再做突围判定（仅用地面投影距离）
        if self.scenario == 'S3' and self.safe_center is not None and self.safe_radius is not None:
            for i in np.where(self.red_alive)[0]:
                pos_xy = np.array([self.red_units[i].X, self.red_units[i].Y], dtype=float)
                center_xy = np.array(self.safe_center[:2], dtype=float)
                dist_xy = np.linalg.norm(pos_xy - center_xy)
                if dist_xy >= float(self.safe_radius):
                    self.red_escaped[i] = True
                    self.red_alive[i] = False
        # 同步数组表示
        self._sync_arrays_from_units()
        # combat
        self._handle_combat()
        # obs/reward/done
        obs = self._get_observation()
        reward = self._compute_reward(obs)
        done, winner = self._check_done()
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
        """单智能体动作映射为每个蓝方单元的 vel 与阵型缩放"""
        alive_indices = np.where(self.blue_alive)[0]
        if len(alive_indices) == 0:
            return
        # compute current formation center
        blue_center = np.mean(self.blue_positions[self.blue_alive], axis=0)
        # discrete legacy actions (int)
        if isinstance(action, (int, np.integer)):
            # 保留原始行为（瞬移式）以向后兼容
            # 这里我们用 position jump（legacy），并把 velocities 清零
            red_center = np.mean(self.red_positions[self.red_alive], axis=0) if np.any(self.red_alive) else blue_center
            vec_to_enemy = red_center - blue_center
            vec2d = np.array([vec_to_enemy[0], vec_to_enemy[1], 0.0])
            if np.linalg.norm(vec2d) < 1e-6:
                direction = np.array([0.0, 0.0, 0.0])
            else:
                direction = vec2d / np.linalg.norm(vec2d)
            left_dir = np.cross(np.array([0.0, 0.0, 1.0]), direction)
            if np.linalg.norm(left_dir) < 1e-6:
                left_dir = np.array([0.0, 0.0, 0.0])
            else:
                left_dir = left_dir / np.linalg.norm(left_dir)
            move_step = 10.0
            if action == 0:
                delta = direction * move_step
            elif action == 1:
                delta = -direction * move_step
            elif action == 2:
                delta = left_dir * move_step
            elif action == 3:
                delta = -left_dir * move_step
            elif action == 4:
                delta = np.array([0.0, 0.0, 10.0])
            elif action == 5:
                delta = np.array([0.0, 0.0, -10.0])
            elif action == 6:
                # gather: radial contraction
                factor = 0.8
                for i in alive_indices:
                    self.blue_units[i].X = blue_center[0] + (self.blue_units[i].X - blue_center[0]) * factor
                    self.blue_units[i].Y = blue_center[1] + (self.blue_units[i].Y - blue_center[1]) * factor
                    self.blue_units[i].Z = blue_center[2] + (self.blue_units[i].Z - blue_center[2]) * factor
                delta = np.zeros(3)
            elif action == 7:
                factor = 1.2
                for i in alive_indices:
                    self.blue_units[i].X = blue_center[0] + (self.blue_units[i].X - blue_center[0]) * factor
                    self.blue_units[i].Y = blue_center[1] + (self.blue_units[i].Y - blue_center[1]) * factor
                    self.blue_units[i].Z = blue_center[2] + (self.blue_units[i].Z - blue_center[2]) * factor
                delta = np.zeros(3)
            else:
                delta = np.zeros(3)
            # apply instantaneous delta and clear velocities
            for i in alive_indices:
                self.blue_units[i].X += delta[0]
                self.blue_units[i].Y += delta[1]
                self.blue_units[i].Z += delta[2]
                self.blue_units[i].vel = np.zeros(3)
            # 同步后直接返回
            self._sync_arrays_from_units()
            return

        # continuous action: interpret as [dx,dy,dz,(formation_scale)]
        act = np.array(action, dtype=float)
        if act.size < 3:
            return
        dir_vec = act[:3]
        formation_scale = float(act[3]) if act.size >= 4 else 1.0
        formation_scale = np.clip(formation_scale, 0.5, 1.5)
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-6:
            desired_dir = np.zeros(3)
        else:
            desired_dir = dir_vec / norm
        desired_velocity = desired_dir * self.blue_max_speed
        # apply formation scaling first (缩放当前阵型位置）
        for i in alive_indices:
            u = self.blue_units[i]
            radial = np.array([u.X, u.Y, u.Z]) - blue_center
            new_pos = blue_center + radial * formation_scale
            u.X, u.Y, u.Z = new_pos.tolist()
            # set velocity
            u.vel = desired_velocity.copy()
        # 更新同步数组
        self._sync_arrays_from_units()

    def _apply_enemy_behavior(self):
        """用速度向量为红方设定 vel（与旧逻辑一致但用 vel*dt 积分）"""
        if self.scenario == 'S1':
            if not np.any(self.blue_alive):
                return
            blue_center = np.mean(self.blue_positions[self.blue_alive], axis=0)
            for i in np.where(self.red_alive)[0]:
                direction = blue_center - self.red_positions[i]
                direction[2] = 0.0
                dist = np.linalg.norm(direction)
                if dist > 1e-6:
                    v = (direction / dist) * self.red_max_speed
                else:
                    v = np.zeros(3)
                self.red_units[i].vel = v
        elif self.scenario == 'S2':
            if not np.any(self.blue_alive):
                return
            blue_center = np.mean(self.blue_positions[self.blue_alive], axis=0)
            recon_alive = False
            recon_pos = None
            if self.blue_recon_index is not None and self.blue_recon_index < len(self.blue_alive) and self.blue_alive[self.blue_recon_index]:
                recon_alive = True
                recon_pos = self.blue_positions[self.blue_recon_index]
            for i in np.where(self.red_alive)[0]:
                if self.red_types[i] == 'J':
                    target = recon_pos if (recon_alive and recon_pos is not None) else blue_center
                    direction = target - self.red_positions[i]
                else:
                    direction = blue_center - self.red_positions[i]
                direction[2] = 0.0
                dist = np.linalg.norm(direction)
                if dist > 1e-6:
                    v = (direction / dist) * self.red_max_speed
                else:
                    v = np.zeros(3)
                self.red_units[i].vel = v
        elif self.scenario == 'S3':
            if self.safe_center is None or self.safe_radius is None:
                return
            for i in np.where(self.red_alive)[0]:
                if not self.red_escaped[i]:
                    # 向安全区外沿径向外逃（在地面投影平面内）
                    direction = self.red_positions[i] - self.safe_center
                    direction[2] = 0.0
                    dist = np.linalg.norm(direction)
                    if dist < 1e-6:
                        direction = np.array([1.0, 0.0, 0.0])
                        dist = 1.0
                    v = (direction / dist) * self.red_max_speed
                    self.red_units[i].vel = v
            # 注意：不在此处预判突围，改为在 step() 位置积分后以二维距离判定
        elif self.scenario == 'S4':
            if not np.any(self.blue_alive):
                return
            blue_center = np.mean(self.blue_positions[self.blue_alive], axis=0)
            decoy_idx = self.red_decoy_index
            for i in np.where(self.red_alive)[0]:
                if i == decoy_idx:
                    dist = np.linalg.norm(self.red_positions[i] - blue_center)
                    if dist < self.decoy_threshold:
                        direction = self.red_positions[i] - blue_center
                        if np.linalg.norm(direction) < 1e-6:
                            self.red_units[i].vel = np.zeros(3)
                        else:
                            self.red_units[i].vel = (direction / np.linalg.norm(direction)) * self.red_max_speed
                    else:
                        self.red_units[i].vel = np.zeros(3)
                else:
                    direction = blue_center - self.red_positions[i]
                    direction[2] = 0.0
                    dist = np.linalg.norm(direction)
                    if dist > 1e-6:
                        self.red_units[i].vel = (direction / dist) * self.red_max_speed
                    else:
                        self.red_units[i].vel = np.zeros(3)
        else:
            return

    def _handle_combat(self):
        """按距离判定击杀，使用 platform 对象的当前位置信息进行判定"""
        blue_alive_indices = np.where(self.blue_alive)[0]
        red_alive_indices = np.where(self.red_alive)[0]
        if len(blue_alive_indices) == 0 or len(red_alive_indices) == 0:
            return
        blue_positions_alive = self.blue_positions[blue_alive_indices]
        red_positions_alive = self.red_positions[red_alive_indices]
        diff = blue_positions_alive[:, np.newaxis, :] - red_positions_alive[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        pairs = []
        for ib in range(dist_matrix.shape[0]):
            for ir in range(dist_matrix.shape[1]):
                d = dist_matrix[ib, ir]
                pairs.append((d, blue_alive_indices[ib], red_alive_indices[ir]))
        pairs.sort(key=lambda x: x[0])
        weapon_range = float(self.weapon_range)
        max_kill_prob_scale = 0.9
        blue_favor = float(self.blue_advantage)
        for d, b_idx, r_idx in pairs:
            if not (self.blue_alive[b_idx] and self.red_alive[r_idx]):
                continue
            if d >= weapon_range:
                continue
            proximity = max(0.0, 1.0 - (d / weapon_range))
            kill_total_prob = max_kill_prob_scale * proximity
            if np.random.rand() < kill_total_prob:
                if np.random.rand() < blue_favor:
                    self.red_alive[r_idx] = False
                    self.blue_kills += 1
                else:
                    self.blue_alive[b_idx] = False
                    self.red_kills += 1
        # 同步可能的 alive/pos 变化
        self._sync_arrays_from_units()

    def _get_observation(self):
        """返回与旧版本兼容的 obs 向量（相对位置，alive 比例等）"""
        if np.any(self.blue_alive) and np.any(self.red_alive):
            blue_center = np.mean(self.blue_positions[self.blue_alive], axis=0)
            red_center = np.mean(self.red_positions[self.red_alive], axis=0)
            relative_pos = blue_center - red_center
        else:
            relative_pos = np.array([0.0, 0.0, 0.0])
        own_alive_ratio = np.sum(self.blue_alive) / len(self.blue_alive)
        enemy_alive_ratio = np.sum(self.red_alive) / len(self.red_alive)
        avg_height_level = 0.0
        if np.any(self.blue_alive) and np.any(self.red_alive):
            diff = self.blue_positions[self.blue_alive][:, np.newaxis, :] - self.red_positions[self.red_alive][np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            nearest_enemy_dist = float(np.min(dists))
        else:
            nearest_enemy_dist = 0.0
        interference_level = 0.0
        if self.scenario == 'S2':
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
        bait_confidence = 0.0
        if self.scenario == 'S4':
            if self.red_decoy_index is not None and self.red_decoy_index < len(self.red_alive) and self.red_alive[self.red_decoy_index]:
                if np.any(self.blue_alive):
                    blue_center = np.mean(self.blue_positions[self.blue_alive], axis=0)
                    decoy_pos = self.red_positions[self.red_decoy_index]
                    dist = np.linalg.norm(decoy_pos - blue_center)
                    bait_confidence = 1.0 if dist < self.decoy_threshold else 0.0
            else:
                bait_confidence = 0.0
        goal_progress = self._compute_goal_progress()
        obs = np.concatenate([
            relative_pos.astype(float),
            [own_alive_ratio, enemy_alive_ratio, avg_height_level, nearest_enemy_dist, interference_level, bait_confidence, goal_progress]
        ]).astype(float)
        return obs

    def _compute_goal_progress(self):
        """与旧逻辑保持一致"""
        progress = 0.0
        if self.scenario == 'S1':
            initial_red = self.num_red_init
            eliminated = self.blue_kills
            progress = eliminated / initial_red
        elif self.scenario == 'S2':
            initial_red = self.num_red_init
            kill_frac = self.blue_kills / initial_red
            recon_alive = 1.0 if (self.blue_recon_index is not None and self.blue_recon_index < self.num_blue_init and self.blue_alive[self.blue_recon_index]) else 0.0
            progress = 0.5 * kill_frac + 0.5 * recon_alive
        elif self.scenario == 'S3':
            esc = np.sum(self.red_escaped)
            progress = 1.0 - min(1.0, esc / 3.0)
        elif self.scenario == 'S4':
            real_indices = [i for i, t in enumerate(self.red_types) if t != 'D']
            real_total = len(real_indices)
            real_alive = sum(1 for i in real_indices if i < len(self.red_alive) and self.red_alive[i])
            real_killed_count = real_total - real_alive
            progress = real_killed_count / real_total if real_total > 0 else 0.0
            if self.red_decoy_index is not None:
                if self.red_decoy_index < len(self.red_alive) and not self.red_alive[self.red_decoy_index] and real_alive > 0:
                    progress -= 0.5
                    if progress < 0:
                        progress = 0.0
        else:
            progress = 0.0
        return progress

    def _compute_reward(self, obs):
        current_kill_diff = self.blue_kills - self.red_kills
        kill_diff_increment = current_kill_diff - self.prev_kill_diff
        self.prev_kill_diff = current_kill_diff
        current_progress = obs[-1]
        progress_change = current_progress - self.prev_progress
        self.prev_progress = current_progress
        blue_alive_ratio = obs[3]
        red_alive_ratio = obs[4]
        survival_diff = blue_alive_ratio - red_alive_ratio
        current_escaped = np.sum(self.red_escaped)
        escape_increment = int(current_escaped - self.prev_escaped_count)
        self.prev_escaped_count = current_escaped
        escape_penalty_term = - self.escape_penalty * escape_increment if escape_increment > 0 else 0.0
        time_penalty = 1.0
        reward = (self.alpha * kill_diff_increment) + (self.beta * progress_change) + (self.gamma * survival_diff) + escape_penalty_term - (self.delta * time_penalty)
        return reward

    def _check_done(self):
        winner = None
        if self.step_count >= self.max_steps:
            if self.scenario == 'S1':
                if self.blue_kills > self.red_kills:
                    winner = 'blue'
                elif self.red_kills > self.blue_kills:
                    winner = 'red'
                else:
                    winner = 'draw'
            elif self.scenario == 'S2':
                blue_score = self.blue_kills + (1 if (self.blue_recon_index is not None and self.blue_recon_index < len(self.blue_alive) and self.blue_alive[self.blue_recon_index]) else 0)
                red_score = self.red_kills
                if blue_score > red_score:
                    winner = 'blue'
                elif red_score > blue_score:
                    winner = 'red'
                else:
                    winner = 'draw'
            elif self.scenario == 'S3':
                if np.sum(self.red_escaped) >= 3:
                    winner = 'red'
                else:
                    winner = 'blue'
            elif self.scenario == 'S4':
                real_indices = [i for i, t in enumerate(self.red_types) if t != 'D']
                any_real_alive = any(i < len(self.red_alive) and self.red_alive[i] for i in real_indices)
                if any_real_alive:
                    winner = 'red'
                else:
                    winner = 'blue'
            else:
                winner = 'draw'
            return True, winner
        blue_alive_count = np.sum(self.blue_alive)
        red_alive_count = np.sum(self.red_alive)
        if blue_alive_count == 0:
            winner = 'red'
            return True, winner
        if red_alive_count == 0:
            if self.scenario == 'S3':
                if np.sum(self.red_escaped) < 3:
                    winner = 'blue'
                else:
                    winner = 'red'
            else:
                winner = 'blue'
            return True, winner
        if self.scenario == 'S3':
            if np.sum(self.red_escaped) >= 3:
                winner = 'red'
                return True, winner
        if self.scenario == 'S4':
            real_indices = [i for i, t in enumerate(self.red_types) if t != 'D']
            all_reals_dead = all((i < len(self.red_alive) and not self.red_alive[i]) for i in real_indices)
            if all_reals_dead:
                winner = 'blue'
                return True, winner
        return False, winner

    def render(self, mode='3d'):
        if mode != '3d':
            print("Only 3D mode is supported for rendering.")
            return
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        blue_alive = np.where(self.blue_alive)[0]
        red_alive = np.where(self.red_alive)[0]
        if len(blue_alive) > 0:
            ax.scatter(self.blue_positions[blue_alive,0], self.blue_positions[blue_alive,1], self.blue_positions[blue_alive,2], c='b', label='Blue')
        if len(red_alive) > 0:
            if self.scenario == 'S4' and self.red_decoy_index is not None and self.red_decoy_index in red_alive:
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