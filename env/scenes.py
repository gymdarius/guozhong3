import numpy as np

def get_scene_config(name):
    """Return configuration dictionary for the specified scenario (S1, S2, S3, S4)."""
    config = {"name": name}
    if name == 'S1':
        # Symmetrical 5v5 confrontation
        # 加大初始间距：蓝方整体左移到 x≈[-60,-40]，红方保持 x≈[100,120]
        blue_positions = np.array([
            [-60.0, 0.0, 50.0],
            [-40.0, 0.0, 50.0],
            [-50.0, -10.0, 50.0],
            [-50.0, 10.0, 50.0],
            [-50.0, 0.0, 50.0]
        ])
        red_positions = np.array([
            [100.0, 0.0, 50.0],
            [120.0, 0.0, 50.0],
            [110.0, -10.0, 50.0],
            [110.0, 10.0, 50.0],
            [110.0, 0.0, 50.0]
        ])
        config['blue_init'] = blue_positions
        config['red_init'] = red_positions
        config['blue_types'] = ['F'] * 5   # F = fighter
        config['red_types'] = ['F'] * 5
        config['weapon_range'] = 30.0

    elif name == 'S2':
        # Heterogeneous confrontation: Blue 4 fighters + 1 recon, Red 3 fighters + 2 jammers
        # 加大初始间距：蓝方整体左移到 x≈[-70,-50]，红方保持 x≈[100,120]
        blue_positions = np.array([
            [-60.0, 0.0, 50.0],    # fighter
            [-55.0, 5.0, 50.0],    # fighter
            [-55.0, -5.0, 50.0],   # fighter
            [-50.0, 10.0, 50.0],   # fighter
            [-70.0, 0.0, 50.0]     # recon (behind others)
        ])
        red_positions = np.array([
            [100.0, 0.0, 50.0],    # fighter
            [120.0, 0.0, 50.0],    # fighter
            [110.0, 10.0, 50.0],   # fighter
            [110.0, -10.0, 50.0],  # jammer
            [115.0, 15.0, 50.0]    # jammer
        ])
        config['blue_init'] = blue_positions
        config['red_init'] = red_positions
        config['blue_types'] = ['F', 'F', 'F', 'F', 'R']  # last is Recon
        config['red_types'] = ['F', 'F', 'F', 'J', 'J']   # last two are Jammers
        config['weapon_range'] = 30.0
        config['jammer_range'] = 40.0
        config['blue_recon_index'] = 4
        config['red_jammer_indices'] = [3, 4]

    elif name == 'S3':
        # Encirclement scenario: Red inside Blue circle, Red attempts breakout
        # 加大包围圈半径，增大红蓝初始间距；同时适度增大安全区半径，留出更多对抗时间
        blue_positions = np.array([
            [80.0, 0.0, 50.0],
            [-80.0, 0.0, 50.0],
            [0.0, 80.0, 50.0],
            [0.0, -80.0, 50.0],
            [56.0, 56.0, 50.0]   # ≈ 80 / sqrt(2)
        ])
        red_positions = np.array([
            [0.0, 0.0, 50.0],
            [5.0, 5.0, 50.0],
            [-5.0, -5.0, 50.0],
            [5.0, -5.0, 50.0],
            [-5.0, 5.0, 50.0]
        ])
        config['blue_init'] = blue_positions
        config['red_init'] = red_positions
        config['blue_types'] = ['F'] * 5
        config['red_types'] = ['F'] * 5
        config['weapon_range'] = 30.0
        config['safe_center'] = np.array([0.0, 0.0, 0.0])  # center of safe zone (project on ground)
        config['safe_radius'] = 60.0  # 原先 50.0 -> 60.0，配合更大包围圈

    elif name == 'S4':
        # Decoy scenario: Red has one decoy and 4 real units
        # 加大初始间距：蓝方整体左移到 x≈[-55,-45]，红方保持 x≈[90,120]
        blue_positions = np.array([
            [-55.0, 0.0, 50.0],
            [-45.0, 0.0, 50.0],
            [-50.0, -5.0, 50.0],
            [-50.0, 5.0, 50.0],
            [-50.0, 0.0, 50.0]
        ])
        red_positions = np.array([
            [90.0, 0.0, 50.0],     # decoy
            [110.0, 0.0, 50.0],    # real
            [120.0, 10.0, 50.0],   # real
            [120.0, -10.0, 50.0],  # real
            [100.0, -5.0, 50.0]    # real
        ])
        config['blue_init'] = blue_positions
        config['red_init'] = red_positions
        config['blue_types'] = ['F'] * 5
        config['red_types'] = ['D', 'F', 'F', 'F', 'F']  # first is Decoy
        config['weapon_range'] = 30.0
        config['red_decoy_index'] = 0
        config['decoy_threshold'] = 30.0  # distance threshold for Blue to identify decoy
    else:
        raise ValueError(f"Unknown scenario {name}")
    return config