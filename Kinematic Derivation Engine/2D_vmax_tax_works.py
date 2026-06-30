import numpy as np
import matplotlib.pyplot as plt

class PureFluidCore2D:
    def __init__(self, name, position, velocity, core_radius=2.0, c_ceiling=1.2):
        self.name = name
        self.pos = np.array(position, dtype=float)  # [x, y]
        self.vel = np.array(velocity, dtype=float)  # [vx, vy]
        self.r = core_radius
        self.c = c_ceiling                          # Saturation speed floor of the medium
        
        # Internal Loop Bedrock Primitives
        self.v_tan = 1.0
        self.v_rad = 1.0

    def get_boundary_unit_vector(self, partner_pos):
        """Calculates the normalized vector pointing from self to partner"""
        vec = partner_pos - self.pos
        dist = np.linalg.norm(vec)
        if dist == 0:
            return np.array([1.0, 0.0]), 0.0
        return vec / dist, dist

def collision_dispersion_2d(v_clash, r_interface, r_node=0.25):
    """
    Sub-Scale 3D Dispersal Function (Level N-1 Vacuum Matrix).
    Dilutes excess clashing velocity by the cube of the interface radius.
    """
    R = max(r_node, r_interface)
    return v_clash * (r_node / R)**3

def calculate_field_deficit_2d(source_pos, target_pos, K=0.08):
    """Calculates velocity potential deficit at a given coordinate point"""
    R = np.linalg.norm(target_pos - source_pos)
    return K / max(0.1, R)

def run_coupled_saturation_engine(steps=1500):
    # Initialize with the standard offsets and speeds
    core_A = PureFluidCore2D("Core_A", position=[-4.0, 0.3], velocity=[0.005, -0.012])
    core_B = PureFluidCore2D("Core_B", position=[4.0, -0.3], velocity=[-0.005, 0.012])
    
    P_ambient = 1.0
    history = {"pos_A": [], "pos_B": [], "distance": [], "v_tax_leeched": []}
    
    for step in range(steps):
        u_hat, distance = core_A.get_boundary_unit_vector(core_B.pos)
        touching_threshold = core_A.r + core_B.r  # 4.0
        r_interface = distance / 2.0
        
        # 1. Map Inner and Outer boundary edge coordinates
        x_in_A = core_A.pos + (u_hat * core_A.r)
        x_out_A = core_A.pos - (u_hat * core_A.r)
        x_in_B = core_B.pos - (u_hat * core_B.r)
        x_out_B = core_B.pos + (u_hat * core_B.r)
        
        # 2. Solve edge pressures via field deficits
        P_out_A = P_ambient - calculate_field_deficit_2d(core_B.pos, x_out_A)
        P_in_A = P_ambient - calculate_field_deficit_2d(core_B.pos, x_in_A)
        P_out_B = P_ambient - calculate_field_deficit_2d(core_A.pos, x_out_B)
        P_in_B = P_ambient - calculate_field_deficit_2d(core_A.pos, x_in_B)
        
        # 3. Apply long-range gravity acceleration from substrate pressure imbalance
        accel_A = (P_out_A - P_in_A) * u_hat * 0.1
        accel_B = (P_out_B - P_in_B) * (-u_hat) * 0.1
        core_A.vel += accel_A
        core_B.vel += accel_B
        
        # 4. Connected Saturation & Dispersion Ledger
        v_tax_total = 0.0
        if distance <= touching_threshold:
            v_relative = core_B.vel - core_A.vel
            closing_speed = np.dot(v_relative, u_hat)
            
            # Process tax only if particles are compressing the medium head-on
            if closing_speed < 0:
                # Evaluate the total combined velocity vector budget against 'c'
                v_macro_sq_A = np.sum(core_A.vel**2)
                v_total_A = np.sqrt(core_A.v_tan**2 + core_A.v_rad**2 + v_macro_sq_A)
                
                if v_total_A > core_A.c:
                    # Calculate the raw velocity excess attempting to cross the ceiling
                    v_excess = v_total_A - core_A.c
                    
                    # Run the excess through the 1/R^3 volumetric dilution profile
                    v_tax_total = collision_dispersion_2d(v_excess, r_interface, r_node=0.45)
                    
                    # Deduct the tax directionally along the line of contact.
                    # Pushes backward against the incoming velocity component.
                    core_A.vel -= (v_tax_total * 0.5) * u_hat
                    core_B.vel += (v_tax_total * 0.5) * u_hat

        # 5. Translate Coordinates
        core_A.pos += core_A.vel
        core_B.pos += core_B.vel
        
        # Telemetry logging
        history["pos_A"].append(core_A.pos.copy())
        history["pos_B"].append(core_B.pos.copy())
        history["distance"].append(distance)
        history["v_tax_leeched"].append(v_tax_total)
        
    return history

# --- Render 2D Performance Dashboard ---
sim_data = run_coupled_saturation_engine()
pos_A = np.array(sim_data["pos_A"])
pos_B = np.array(sim_array := sim_data["pos_B"])

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: 2D Spatial Tracks
axs[0].plot(pos_A[:, 0], pos_A[:, 1], color='blue', linewidth=2, label='Core A Track')
axs[0].plot(pos_B[:, 0], pos_B[:, 1], color='red', linewidth=2, label='Core B Track')
axs[0].scatter(pos_A[0, 0], pos_A[0, 1], color='blue', marker='o', s=100, label='A Start')
axs[0].scatter(pos_B[0, 0], pos_B[0, 1], color='red', marker='o', s=100, label='B Start')
axs[0].scatter(pos_A[-1, 0], pos_A[-1, 1], color='purple', marker='X', s=150, label='Stable Settlement')
axs[0].scatter(pos_B[-1, 0], pos_B[-1, 1], color='purple', marker='X', s=150)
axs[0].set_title("2D Connected Saturation-Dispersion Trajectory", fontsize=12, fontweight='bold')
axs[0].set_xlabel("X Coordinate Space", fontweight='bold')
axs[0].set_ylabel("Y Coordinate Space", fontweight='bold')
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].axis('equal')
axs[0].legend()

# Chart 2: Distance and Dispersion Tax Tracking
axs[1].plot(sim_data["distance"], color='green', linewidth=2.5, label='Separation ($R_{AB}$)')
axs[1].plot(sim_data["v_tax_leeched"], color='orange', linestyle='--', linewidth=2, label='Substrate Volumetric Tax ($v_{tax}$)')
axs[1].axhline(4.0, color='black', linestyle=':', label='Touch Horizon (4.0)')
axs[1].set_title("Dynamic Settlement Ledger", fontsize=12, fontweight='bold')
axs[1].set_xlabel("Simulation Steps", fontweight='bold')
axs[1].set_ylabel("Metrics Magnitude", fontweight='bold')
axs[1].grid(True, linestyle='--', alpha=0.5)
axs[1].legend()

plt.tight_layout()
plt.show()