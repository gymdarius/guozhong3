# class/platforms.py
# 新增：面向不同机种的继承类（Recon, Fighter, Jammer, Decoy）
# 依赖：class/trajectory.py 中的 Aircraft, Missiles

import numpy as np
import math as m

from flat_models.trajectory import Aircraft, Missiles

class Recon(Aircraft):
    """侦察机：更强的传感能力，但火力弱或无火力"""
    def __init__(self, aircraft_plist, V, Pitch, Heading, sensor_range=200.0, detect_prob=0.9, **kwargs):
        super().__init__(aircraft_plist, V, Pitch, Heading, **kwargs)
        self.sensor_range = float(sensor_range)
        self.detect_prob = float(detect_prob)
        # 我们为平台对象增加一个便捷的速度向量属性（用于 kinematic 更新）
        self.vel = np.zeros(3, dtype=float)

    def detect(self, targets_positions):
        """
        给出可探测目标索引和探测概率（简单模型：距离小于sensor_range并按概率探测）
        targets_positions: list or ndarray of positions [[x,y,z], ...]
        返回：list of booleans 或 detected positions
        """
        detected = []
        for p in targets_positions:
            d = np.linalg.norm(np.array(p) - np.array([self.X, self.Y, self.Z]))
            if d <= self.sensor_range:
                detected.append(np.random.rand() < self.detect_prob)
            else:
                detected.append(False)
        return detected

class Fighter(Aircraft):
    """歼击机：高机动、武器（近距离）"""
    def __init__(self, aircraft_plist, V, Pitch, Heading, weapon_range=30.0, attack_power=1.0, **kwargs):
        super().__init__(aircraft_plist, V, Pitch, Heading, **kwargs)
        self.weapon_range = float(weapon_range)
        self.attack_power = float(attack_power)
        self.vel = np.zeros(3, dtype=float)

    def can_attack(self, target_pos):
        d = np.linalg.norm(np.array(target_pos) - np.array([self.X, self.Y, self.Z]))
        return d <= self.weapon_range

    def attack_success(self, target_pos, base_prob=0.5):
        """
        简单攻击命中概率模型：基于距离（越近命中概率越高）和attack_power
        """
        d = np.linalg.norm(np.array(target_pos) - np.array([self.X, self.Y, self.Z]))
        if d > self.weapon_range:
            return False
        proximity = 1.0 - (d / self.weapon_range)
        prob = base_prob * (0.5 + 0.5 * proximity) * self.attack_power
        return np.random.rand() < min(1.0, prob)

class Jammer(Aircraft):
    """干扰机：对一定半径内友方侦察造成干扰（环境会读取 jam_strength）"""
    def __init__(self, aircraft_plist, V, Pitch, Heading, jam_range=40.0, jam_strength=1.0, **kwargs):
        super().__init__(aircraft_plist, V, Pitch, Heading, **kwargs)
        self.jam_range = float(jam_range)
        self.jam_strength = float(jam_strength)
        self.vel = np.zeros(3, dtype=float)

    def is_in_jam(self, point):
        d = np.linalg.norm(np.array(point) - np.array([self.X, self.Y, self.Z]))
        return d <= self.jam_range

class Decoy(Aircraft):
    """诱饵机：低成本、用来吸引敌方，可能无武器，但signature_level高（更易被判定为重要目标）"""
    def __init__(self, aircraft_plist, V, Pitch, Heading, signature_level=1.0, lure_strength=1.0, **kwargs):
        super().__init__(aircraft_plist, V, Pitch, Heading, **kwargs)
        self.signature_level = float(signature_level)
        self.lure_strength = float(lure_strength)
        self.vel = np.zeros(3, dtype=float)

    def lure_behavior(self, blue_center):
        """
        简单策略：诱饵会向蓝方靠近或保持一定姿态以吸引攻击（可在环境脚本中使用）
        """
        direction = np.array(blue_center) - np.array([self.X, self.Y, self.Z])
        if np.linalg.norm(direction) < 1e-6:
            return
        direction = direction / np.linalg.norm(direction)
        move_v = 5.0 * self.lure_strength
        self.X += direction[0] * move_v * self.dt
        self.Y += direction[1] * move_v * self.dt
        self.Z += direction[2] * move_v * self.dt