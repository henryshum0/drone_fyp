import numpy as np
from gym_pybullet_drones.gateRL.interpolate import interpolate_waypoints
from transforms3d.euler import euler2quat, quat2euler

# Original simple square
waypoints1 = {
    "pos":np.array([
        np.array([0, 0, 0]),
        np.array([4, 0, 0]),
        np.array([4, 4, 0]),
        np.array([0, 4, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi/2]),
        np.array([0, 0, np.pi]),
        np.array([0, 0, np.pi * 3/2]),
    ]),
    "spawn":[
        {
            "pos": np.array([-.4, .4, -.4]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 6.,
}

# Figure-8 pattern (3D)
waypoints_figure8 = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([2, 2, 1]),
        np.array([4, 0, 2]),
        np.array([2, -2, 1]),
        np.array([0, 0, 0]),
        np.array([-2, 2, 1]),
        np.array([-4, 0, 2]),
        np.array([-2, -2, 1]),
    ]),
    "rpy": np.array([
        np.array([0, 0, np.pi/4]),
        np.array([0, 0.1, 0.]),
        np.array([0, 0.2, np.pi* -1/2]),
        np.array([0, 0.1, np.pi]),
        np.array([0, 0, np.pi * 3/4]),
        np.array([0, 0.1, np.pi]),
        np.array([0, 0.2, np.pi * -1 /2 ]),
        np.array([0, 0.1, 0.]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1.1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 4.,
}

# Spiral ascent
waypoints_spiral = {
    "pos": np.array([
        np.array([3 * np.cos(t), 3 * np.sin(t), t * 0.5]) 
        for t in np.linspace(0, 4*np.pi, 16)
    ]),
    "rpy": np.array([
        np.array([0, 0, t + np.pi/2]) 
        for t in np.linspace(0, 4*np.pi, 16)
    ]),
    "spawn": [
        {
            "pos": np.array([3, 0, -0.5]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, np.pi/2]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 10.,
}

# Slalom course (zigzag)
waypoints_slalom = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([2, 2, 0.5]),
        np.array([4, -2, 1]),
        np.array([6, 2, 1.5]),
        np.array([8, -2, 2]),
        np.array([10, 0, 2]),
    ]),
    "rpy": np.array([
        np.array([0, 0, np.pi/4]),
        np.array([0, 0.1, np.pi]),
        np.array([0, 0.1, 0]),
        np.array([0, 0.1, np.pi]),
        np.array([0, 0.1, 0]),
        np.array([0, 0, np.pi/4]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, np.pi/4]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 12.,
}

# Vertical loop
waypoints_loop = {
    "pos": np.array([
        np.array([0, 0, 0]),
        np.array([2, 0, 2]),
        np.array([4, 0, 4]),
        np.array([6, 0, 2]),
        np.array([8, 0, 0]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, np.pi/4, 0]),
        np.array([0, np.pi/2, 0]),
        np.array([0, np.pi*3/4, 0]),
        np.array([0, np.pi, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1, 0, 0]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 10.,
}

# Sharp turns (aggressive maneuvers)
waypoints_sharp_turns = {
    "pos": np.array([
        np.array([0, 0, 1]),
        np.array([3, 0, 1]),
        np.array([3, 3, 1]),
        np.array([0, 3, 1]),
        np.array([0, 6, 1]),
        np.array([3, 6, 1]),
    ]),
    "rpy": np.array([
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi/2]),  # 90° turn
        np.array([0, 0, np.pi]),    # 90° turn
        np.array([0, 0, np.pi/2]),  # -90° turn
        np.array([0, 0, 0]),        # -90° turn
        np.array([0, 0, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1, 0, 1]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 8.,
}

# High-speed straight with tilted gates
waypoints_speed_run = {
    "pos": np.array([
        np.array([0, 0, 1]),
        np.array([3, 1, 1.5]),
        np.array([6, -1, 2]),
        np.array([9, 1, 2.5]),
        np.array([12, 0, 3]),
    ]),
    "rpy": np.array([
        np.array([0, 0.2, 0.1]),
        np.array([0.1, 0.3, 0.2]),
        np.array([-0.1, 0.3, -0.1]),
        np.array([0.1, 0.2, 0.1]),
        np.array([0, 0.1, 0]),
    ]),
    "spawn": [
        {
            "pos": np.array([-1, 0, 1]),
            "vel": np.array([2, 0, 0]),  # Start with velocity
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0.1, 0]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 15.,
}

# 3D helix
waypoints_helix = {
    "pos": np.array([
        np.array([2 * np.cos(t), 2 * np.sin(t), t * 0.3]) 
        for t in np.linspace(0, 6*np.pi, 20)
    ]),
    "rpy": np.array([
        np.array([0, 0.1, t + np.pi/2]) 
        for t in np.linspace(0, 6*np.pi, 20)
    ]),
    "spawn": [
        {
            "pos": np.array([2, 0, -0.3]),
            "vel": np.array([0, 0, 0]),
            "acc": np.array([0, 0, 0]),
            "rpy": np.array([0, 0, np.pi/2]),
            "next_waypoints": [0, 1]
        }
    ],
    "max_dist": 12.,
}

# Random waypoints generator
def generate_random_waypoints(n_waypoints=8, 
                            bounds=(-5, 5), 
                            height_range=(0.5, 3),
                            max_angle=np.pi/3,
                            seed=None):
    """
    Generate random waypoints for varied training
    
    Args:
        n_waypoints: Number of waypoints
        bounds: (min, max) for x, y coordinates
        height_range: (min, max) for z coordinate
        max_angle: Maximum roll/pitch angle
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    pos = []
    rpy = []
    
    for i in range(n_waypoints):
        x = np.random.uniform(*bounds)
        y = np.random.uniform(*bounds)
        z = np.random.uniform(*height_range)
        pos.append(np.array([x, y, z]))
        
        roll = np.random.uniform(-max_angle/2, max_angle/2)
        pitch = np.random.uniform(-max_angle/2, max_angle/2)
        yaw = np.random.uniform(-np.pi, np.pi)
        rpy.append(np.array([roll, pitch, yaw]))
    
    return {
        "pos": np.array(pos),
        "rpy": np.array(rpy),
        "spawn": [
            {
                "pos": pos[0] - np.array([1, 0, 0]),
                "vel": np.array([0, 0, 0]),
                "acc": np.array([0, 0, 0]),
                "rpy": rpy[0],
                "next_waypoints": [0, 1]
            }
        ],
        "max_dist": np.max(np.linalg.norm(np.diff(pos, axis=0), axis=1)) * 2,
    }

# Collection of all waypoint sets (easy to hard)
ALL_WAYPOINTS = {
    "square": waypoints1,
    "figure8": waypoints_figure8,
    "slalom": waypoints_slalom,
    "sharp_turns": waypoints_sharp_turns,
    "loop": waypoints_loop,
    "spiral": waypoints_spiral,
    "speed_run": waypoints_speed_run,
    "helix": waypoints_helix,
}

# Generate 5 random courses for variety
for i in range(5):
    ALL_WAYPOINTS[f"random_{i}"] = generate_random_waypoints(
        n_waypoints=np.random.randint(6, 12),
        seed=42 + i
    )

# Old waypoints2 format
waypoints2 = np.array([
    [
        np.array([2, 0, 0]),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([-1, 0, 0]),
        np.array([0, -1, 0]),
        np.array([1, 0, 0]),
    ],
    [
        np.array([0, 0, np.pi]),
        np.array([0, 0, np.pi/4 * 3]),
        np.array([0, 0, np.pi]),
        np.array([0, 0, -np.pi/2]),
        np.array([0, 0, 0]),
        np.array([0, 0, np.pi/2]),
    ]
])
