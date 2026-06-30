import numpy as np
import matplotlib.pyplot as plt

class KinematicCore2D:
    def __init__(self, name, position, velocity, core_radius=2.0):
        self.name = name
        self.pos = np.array(position, dtype=float)  # [x, y]
        self.vel = np.array(velocity, dtype=float)  # [vx, vy]
        self.r = core_radius
        self.is_pinned = False

    def get_boundary_unit_vector(self, partner_pos):
        """Calculates the normalized vector pointing from self to partner"""
        vec = partner_pos - self.pos
        dist = np.linalg.norm(vec)
        if dist == 0:
            return np.array([1.0, 0.0]), 0.0
        return vec / dist, dist

def calculate_field_deficit_2d(source_pos, target_pos, K=0.08):
    """Calculates velocity potential deficit at a 2D coordinate point"""
    R = np.linalg.norm(target_pos - source_pos)
    return K / max(0.1, R)

def run_2d_pressure_engine(steps=1200):
    # Initialize cores in a 2D plane with tangential velocities for orbit tracking
    core_A = KinematicCore2D("Core_A", position=[-4.0, 0.3], velocity=[0.005, -0.012])
    core_B = KinematicCore2D("Core_B", position=[4.0, -0.3], velocity=[-0.005, 0.012])
    
    P_ambient = 1.0
    history = {"pos_A": [], "pos_B": [], "distance": [], "P_delta_A": []}
    
    for step in range(steps):
        # 1. Get relative axis geometry and macro separation distance
        u_hat, distance = core_A.get_boundary_unit_vector(core_B.pos)
        touching_threshold = core_A.r + core_B.r  # 4.0
        
        # 2. Check for Harmonic Resonance Lock-In
        if not core_A.is_pinned and distance <= touching_threshold:
            core_A.is_pinned = True
            core_B.is_pinned = True
            core_A.vel = np.array([0.0, 0.0])
            core_B.vel = np.array([0.0, 0.0])
            
            # Lock coordinates neatly on the contact boundary vector
            midpoint = (core_A.pos + core_B.pos) / 2.0
            core_A.pos = midpoint - u_hat * (touching_threshold / 2.0)
            core_B.pos = midpoint + u_hat * (touching_threshold / 2.0)

        # 3. Dynamic Kinematics Block
        if not core_A.is_pinned:
            # Map Inner and Outer boundary edge coordinates in 2D space
            x_in_A = core_A.pos + (u_hat * core_A.r)
            x_out_A = core_A.pos - (u_hat * core_A.r)
            
            x_in_B = core_B.pos - (u_hat * core_B.r)
            x_out_B = core_B.pos + (u_hat * core_B.r)
            
            # Calculate local substrate pressure at the edges
            P_out_A = P_ambient - calculate_field_deficit_2d(core_B.pos, x_out_A)
            P_in_A = P_ambient - calculate_field_deficit_2d(core_B.pos, x_in_A)
            
            P_out_B = P_ambient - calculate_field_deficit_2d(core_A.pos, x_out_B)
            P_in_B = P_ambient - calculate_field_deficit_2d(core_A.pos, x_in_B)
            
            # Scalar pressure deltas across the diameter of each core
            delta_P_A = P_out_A - P_in_A
            delta_P_B = P_out_B - P_in_B
            
            # The Magic Moment: No direction sign tracking needed. 
            # The pressure imbalance accelerates the core strictly along the unit vector.
            accel_A = delta_P_A * u_hat * 0.1
            accel_B = delta_P_B * (-u_hat) * 0.1
            
            core_A.vel += accel_A
            core_B.vel += accel_B
            
            core_A.pos += core_A.vel
            core_B.pos += core_B.vel
        else:
            # Maintain static pressure logs for the dashboard once locked
            P_out_A = P_ambient - calculate_field_deficit_2d(core_B.pos, core_A.pos - u_hat * core_A.r)
            P_in_A = P_ambient - calculate_field_deficit_2d(core_B.pos, core_A.pos + u_hat * core_A.r)
            delta_P_A = P_out_A - P_in_A

        # Telemetry logging
        history["pos_A"].append(core_A.pos.copy())
        history["pos_B"].append(core_B.pos.copy())
        history["distance"].append(distance)
        history["P_delta_A"].append(delta_P_A)
        
    return history

# --- Render 2D Performance Dashboard ---
sim_data = run_2d_pressure_engine()
pos_A = np.array(sim_data["pos_A"])
pos_B = np.array(sim_data["pos_B"])

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Dashboard Chart 1: Actual Spatial 2D Trajectories
axs[0].plot(pos_A[:, 0], pos_A[:, 1], color='blue', linewidth=2, label='Core A Track')
axs[0].plot(pos_B[:, 0], pos_B[:, 1], color='red', linewidth=2, label='Core B Track')
axs[0].scatter(pos_A[0, 0], pos_A[0, 1], color='blue', marker='o', s=100, label='A Start')
axs[0].scatter(pos_B[0, 0], pos_B[0, 1], color='red', marker='o', s=100, label='B Start')
axs[0].scatter(pos_A[-1, 0], pos_A[-1, 1], color='purple', marker='X', s=150, label='Resonance Lock')
axs[0].scatter(pos_B[-1, 0], pos_B[-1, 1], color='purple', marker='X', s=150)
axs[0].set_title("2D Orbital Trajectory & Geometric Lock", fontsize=12, fontweight='bold')
axs[0].set_xlabel("X Coordinate Space", fontweight='bold')
axs[0].set_ylabel("Y Coordinate Space", fontweight='bold')
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].axis('equal')
axs[0].legend()

# Dashboard Chart 2: Relational Distance Drop Over Time
axs[1].plot(sim_data["distance"], color='green', linewidth=2.5, label='Separation ($R_{AB}$)')
axs[1].axhline(4.0, color='black', linestyle=':', label='Lock Perimeter (4.0)')
axs[1].set_title("Macroscopic Separation Profile", fontsize=12, fontweight='bold')
axs[1].set_xlabel("Simulation Steps", fontweight='bold')
axs[1].set_ylabel("Distance", fontweight='bold')
axs[1].grid(True, linestyle='--', alpha=0.5)
axs[1].legend()

plt.tight_layout()
plt.show()