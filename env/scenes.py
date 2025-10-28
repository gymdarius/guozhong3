import numpy as np

def get_scene_config(name):
    """Return configuration dictionary for the specified scenario (S1, S2, S3, S4)."""
    config = {"name": name}
    if name == 'S1':
        # Symmetrical 5v5 confrontation
        blue_positions = np.array([
            [-10.0, 0.0, 50.0],
            [10.0, 0.0, 50.0],
            [0.0, -10.0, 50.0],
            [0.0, 10.0, 50.0],
            [0.0, 0.0, 50.0]
        ])
        red_positions = np.array([
            [90.0, 0.0, 50.0],
            [110.0, 0.0, 50.0],
            [100.0, -10.0, 50.0],
            [100.0, 10.0, 50.0],
            [100.0, 0.0, 50.0]
        ])
        config['blue_init'] = blue_positions
        config['red_init'] = red_positions
        config['blue_types'] = ['F'] * 5   # F = fighter
        config['red_types'] = ['F'] * 5
        config['weapon_range'] = 30.0
        # No special units
    elif name == 'S2':
        # Heterogeneous confrontation: Blue 4 fighters + 1 recon, Red 3 fighters + 2 jammers
        blue_positions = np.array([
            [10.0, 0.0, 50.0],   # fighter
            [5.0, 5.0, 50.0],    # fighter
            [5.0, -5.0, 50.0],   # fighter
            [0.0, 10.0, 50.0],   # fighter
            [-10.0, 0.0, 50.0]   # recon (behind others)
        ])
        red_positions = np.array([
            [90.0, 0.0, 50.0],   # fighter
            [110.0, 0.0, 50.0],  # fighter
            [100.0, 10.0, 50.0], # fighter
            [100.0, -10.0, 50.0],# jammer
            [105.0, 15.0, 50.0]  # jammer
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
        blue_positions = np.array([
            [50.0, 0.0, 50.0],
            [-50.0, 0.0, 50.0],
            [0.0, 50.0, 50.0],
            [0.0, -50.0, 50.0],
            [35.0, 35.0, 50.0]
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
        config['safe_radius'] = 50.0
    elif name == 'S4':
        # Decoy scenario: Red has one decoy and 4 real units
        blue_positions = np.array([
            [-5.0, 0.0, 50.0],
            [5.0, 0.0, 50.0],
            [0.0, -5.0, 50.0],
            [0.0, 5.0, 50.0],
            [0.0, 0.0, 50.0]
        ])
        red_positions = np.array([
            [80.0, 0.0, 50.0],    # decoy
            [100.0, 0.0, 50.0],   # real
            [110.0, 10.0, 50.0],  # real
            [110.0, -10.0, 50.0], # real
            [90.0, -5.0, 50.0]    # real
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
