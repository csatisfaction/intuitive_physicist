import numpy as np
import matplotlib.pyplot as plt

class ResonanceFluidCore2D:
    def __init__(self, name, position, velocity, core_radius=2.0, c_ceiling=1.2):
        self.name = name
        self.pos = np.array(position, dtype=float)  # [x, y]
        self.vel = np.array(velocity, dtype=float)  # [vx, vy]
        self.r = core_radius
        self.c = c_ceiling                          # Medium propagation velocity limit
        
        # Core Internal Invariant Primitives
        self.v_tan = 1.0
        self.v_rad = 1.0

    def get_boundary_unit_vector(self, partner_pos):
        """Calculates the normalized vector pointing from self to partner"""
        vec = partner_pos - self.pos
        dist = np.linalg.norm(vec)
        if dist == 0:
            return np.array([1.0, 0.0]), 0.0
        return vec / dist, dist

def calculate_field_deficit_2d(source_pos, target_pos, K=0.08):
    """Calculates velocity potential deficit at a given coordinate point"""
    R = np.linalg.norm(target_pos - source_pos)
    return K / max(0.1, R)

def run_resonance_filter_engine(steps=2000):
    # Initialize with the standard structural offsets and starting speeds
    core_A = ResonanceFluidCore2D("Core_A", position=[-4.0, 0.3], velocity=[0.005, -0.012])
    core_B = ResonanceFluidCore2D("Core_B", position=[4.0, -0.3], velocity=[-0.005, 0.012])
    
    P_ambient = 1.0
    lambda_wave = 4.0  # Fundamental harmonic wavelength node matching the perimeters (r_A + r_B)
    r_node = 0.5       # Core sub-scale node boundary scale
    
    history = {"pos_A": [], "pos_B": [], "distance": [], "impedance": []}
    
    for step in range(steps):
        u_hat, distance = core_A.get_boundary_unit_vector(core_B.pos)
        r_interface = distance / 2.0
        
        # 1. Map Inner and Outer boundary edge coordinates
        x_out_A = core_A.pos - (u_hat * core_A.r)
        x_in_A = core_A.pos + (u_hat * core_A.r)
        x_in_B = core_B.pos - (u_hat * core_B.r)
        x_out_B = core_B.pos + (u_hat * core_B.r)
        
        # 2. Solve edge pressures via field deficits
        P_out_A = P_ambient - calculate_field_deficit_2d(core_B.pos, x_out_A)
        P_in_A = P_ambient - calculate_field_deficit_2d(core_B.pos, x_in_A)
        P_out_B = P_ambient - calculate_field_deficit_2d(core_A.pos, x_out_B)
        P_in_B = P_ambient - calculate_field_deficit_2d(core_A.pos, x_in_B)
        
        # 3. Apply emergent gravity acceleration from substrate pressure imbalance
        accel_A = (P_out_A - P_in_A) * u_hat * 0.1
        accel_B = (P_out_B - P_in_B) * (-u_hat) * 0.1
        core_A.vel += accel_A
        core_B.vel += accel_B
        
        # 4. UNIFIED PHASE RESONANCE FILTER (No Hardcoded Directional Triggers)
        # The wave-phase resonance factor evaluates to 0 at whole-integer nodes, and spikes out of phase
        resonance_factor = np.sin(np.pi * distance / lambda_wave)**2
        
        # Volumetric dilution profile (1/R^3 matrix scaling)
        dilution = (r_node / max(r_node, r_interface))**3
        
        # Total continuous medium impedance acting on all motion vectors uniformly
        impedance = resonance_factor * dilution * 0.08
        
        # Drag is applied strictly across the entire velocity vector uniformly
        core_A.vel -= core_A.vel * impedance
        core_B.vel -= core_B.vel * impedance
        
        # Enforce global v=c velocity saturation ceiling budget
        for core in [core_A, core_B]:
            v_total = np.sqrt(core.v_tan**2 + core.v_rad**2 + np.sum(core.vel**2))
            if v_total > core.c:
                core.vel *= (core.c / v_total)

        # 5. Translate Coordinates
        core_A.pos += core_A.vel
        core_B.pos += core_B.vel
        
        # Telemetry logging
        history["pos_A"].append(core_A.pos.copy())
        history["pos_B"].append(core_B.pos.copy())
        history["distance"].append(distance)
        history["impedance"].append(impedance)
        
    return history

# --- Render 2D Performance Dashboard ---
sim_data = run_resonance_filter_engine()
pos_A = np.array(sim_data["pos_A"])
pos_B = np.array(sim_data["pos_B"])

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: 2D Spatial Tracks
axs[0].plot(pos_A[:, 0], pos_A[:, 1], color='blue', linewidth=2, label='Core A Track')
axs[0].plot(pos_B[:, 0], pos_B[:, 1], color='red', linewidth=2, label='Core B Track')
axs[0].scatter(pos_A[0, 0], pos_A[0, 1], color='blue', marker='o', s=100, label='A Start')
axs[0].scatter(pos_B[0, 0], pos_B[0, 1], color='red', marker='o', s=100, label='B Start')
axs[0].scatter(pos_A[-1, 0], pos_A[-1, 1], color='purple', marker='X', s=150, label='Resonance Groove')
axs[0].scatter(pos_B[-1, 0], pos_B[-1, 1], color='purple', marker='X', s=150)
axs[0].set_title("2D Path: True Emergent Phase-Resonance Capture", fontsize=12, fontweight='bold')
axs[0].set_xlabel("X Coordinate Space", fontweight='bold')
axs[0].set_ylabel("Y Coordinate Space", fontweight='bold')
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].axis('equal')
axs[0].legend()

# Chart 2: Relational Distance Drop Over Time
axs[1].plot(sim_data["distance"], color='green', linewidth=2.5, label='Separation ($R_{AB}$)')
axs[1].plot(np.array(sim_data["impedance"]) * 100, color='orange', linestyle='--', linewidth=2, label='Substrate Impedance Profile ($\\times 100$)')
axs[1].axhline(4.0, color='black', linestyle=':', label='Harmonic Node (4.0)')
axs[1].set_title("Dynamic Field Impedance vs. Distance", fontsize=12, fontweight='bold')
axs[1].set_xlabel("Simulation Steps", fontweight='bold')
axs[1].set_ylabel("Metrics Magnitude", fontweight='bold')
axs[1].grid(True, linestyle='--', alpha=0.5)
axs[1].legend()

plt.tight_layout()
plt.show()