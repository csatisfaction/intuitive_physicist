import numpy as np
import matplotlib.pyplot as plt

class KinematicCore2D:
    def __init__(self, name, position, velocity, core_radius=2.0):
        self.name = name
        self.pos = np.array(position, dtype=float)  # [x, y]
        self.vel = np.array(velocity, dtype=float)  # [vx, vy]
        self.r = core_radius

    def get_boundary_unit_vector(self, partner_pos):
        """Calculates the normalized vector pointing from self to partner"""
        vec = partner_pos - self.pos
        dist = np.linalg.norm(vec)
        if dist == 0:
            return np.array([1.0, 0.0]), 0.0
        return vec / dist, dist

def collision_dispersion_2d(v_clash, r_interface, r_node=0.05):
    """
    Sub-Scale 3D Dispersal Function mapped to the 2D interface plane.
    Leeches velocity based on the volumetric density of background nodes.
    """
    R = max(r_node, r_interface)
    return v_clash * (r_node / R)**3

def calculate_field_deficit_2d(source_pos, target_pos, K=0.08):
    """Calculates velocity potential deficit at a given coordinate point"""
    R = np.linalg.norm(target_pos - source_pos)
    return K / max(0.1, R)

def run_pure_dispersion_engine(steps=1500):
    # Initialize cores with the exact same spatial offsets and speeds as before
    core_A = KinematicCore2D("Core_A", position=[-4.0, 0.3], velocity=[0.005, -0.012])
    core_B = KinematicCore2D("Core_B", position=[4.0, -0.3], velocity=[-0.005, 0.012])
    
    P_ambient = 1.0
    history = {"pos_A": [], "pos_B": [], "distance": [], "v_tax_leeched": []}
    
    for step in range(steps):
        # 1. Get relative interaction axis geometry
        u_hat, distance = core_A.get_boundary_unit_vector(core_B.pos)
        touching_threshold = core_A.r + core_B.r  # 4.0
        r_interface = distance / 2.0
        
        # 2. Map Inner and Outer boundary edge coordinates
        x_in_A = core_A.pos + (u_hat * core_A.r)
        x_out_A = core_A.pos - (u_hat * core_A.r)
        x_in_B = core_B.pos - (u_hat * core_B.r)
        x_out_B = core_B.pos + (u_hat * core_B.r)
        
        # 3. Solve edge pressures via field deficits
        P_out_A = P_ambient - calculate_field_deficit_2d(core_B.pos, x_out_A)
        P_in_A = P_ambient - calculate_field_deficit_2d(core_B.pos, x_in_A)
        P_out_B = P_ambient - calculate_field_deficit_2d(core_A.pos, x_out_B)
        P_in_B = P_ambient - calculate_field_deficit_2d(core_A.pos, x_in_B)
        
        # 4. Apply long-range emergent gravity acceleration
        accel_A = (P_out_A - P_in_A) * u_hat * 0.1
        accel_B = (P_out_B - P_in_B) * (-u_hat) * 0.1
        core_A.vel += accel_A
        core_B.vel += accel_B
        
        # 5. THE ORGANIC IMPEDANCE BREAK: Substrate Velocity Tax
        v_tax_total = 0.0
        if distance <= touching_threshold:
            # Isolate the relative velocity vector component along the interaction axis
            v_relative = core_B.vel - core_A.vel
            closing_speed = np.dot(v_relative, u_hat)
            
            # Only tax the particles if they are actively encroaching/clashing inward
            if closing_speed < 0:
                # Run the closing speed through the 1/R^3 volumetric dilution ledger
                v_tax_total = collision_dispersion_2d(abs(closing_speed), r_interface, r_node=0.15)
                
                # Deduct the tax directly from their multi-dimensional velocity vectors
                # This leeches momentum precisely along the line of center-point contact
                core_A.vel += (v_tax_total * 0.5) * u_hat
                core_B.vel -= (v_tax_total * 0.5) * u_hat

        # 6. Translate Coordinates
        core_A.pos += core_A.vel
        core_B.pos += core_B.vel
        
        # Telemetry logging
        history["pos_A"].append(core_A.pos.copy())
        history["pos_B"].append(core_B.pos.copy())
        history["distance"].append(distance)
        history["v_tax_leeched"].append(v_tax_total)
        
    return history

# --- Render 2D Performance Dashboard ---
sim_data = run_pure_dispersion_engine()
pos_A = np.array(sim_data["pos_A"])
pos_B = np.array(sim_data["pos_B"])

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: Actual Spatial 2D Trajectories
axs[0].plot(pos_A[:, 0], pos_A[:, 1], color='blue', linewidth=2, label='Core A Track')
axs[0].plot(pos_B[:, 0], pos_B[:, 1], color='red', linewidth=2, label='Core B Track')
axs[0].scatter(pos_A[0, 0], pos_A[0, 1], color='blue', marker='o', s=100, label='A Start')
axs[0].scatter(pos_B[0, 0], pos_B[0, 1], color='red', marker='o', s=100, label='B Start')
axs[0].scatter(pos_A[-1, 0], pos_A[-1, 1], color='purple', marker='X', s=150, label='Final State')
axs[0].scatter(pos_B[-1, 0], pos_B[-1, 1], color='purple', marker='X', s=150)
axs[0].set_title("2D Unclamped Trajectory: Emergent Substrate Braking", fontsize=12, fontweight='bold')
axs[0].set_xlabel("X Coordinate Space", fontweight='bold')
axs[0].set_ylabel("Y Coordinate Space", fontweight='bold')
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].axis('equal')
axs[0].legend()

# Chart 2: Relational Distance Drop Over Time
axs[1].plot(sim_data["distance"], color='green', linewidth=2.5, label='Separation ($R_{AB}$)')
axs[1].plot(sim_data["v_tax_leeched"], color='red', linestyle=':', linewidth=2, label='Substrate Tax ($v_{tax}$)')
axs[1].axhline(4.0, color='black', linestyle='--', alpha=0.7, label='Touch Horizon (4.0)')
axs[1].set_title("Dynamic Separation & Energy Handoff", fontsize=12, fontweight='bold')
axs[1].set_xlabel("Simulation Steps", fontweight='bold')
axs[1].set_ylabel("Metrics Magnitude", fontweight='bold')
axs[1].grid(True, linestyle='--', alpha=0.5)
axs[1].legend()

plt.tight_layout()
plt.show()