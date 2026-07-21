import numpy as np
import matplotlib.pyplot as plt

class PureKinematicCore2D:
    def __init__(self, name, position, velocity, spin_magnitude=1.5, radius=1.5, mass=1.0):
        self.name = name
        self.pos = np.array(position, dtype=float)   # [x, y]
        self.vel = np.array(velocity, dtype=float)   # [vx, vy]
        self.spin = float(spin_magnitude)            # Gyroscopic out-of-plane momentum (Z-axis)
        self.r = float(radius)
        self.mass = float(mass)

def get_interaction_geometry(self, partner_pos):
        """Calculates separation axis vector and true physical gap distance"""
        vec = partner_pos - self.pos
        distance = np.linalg.norm(vec)
        if distance == 0:
            return np.array([1.0, 0.0]), 0.0, 0.0
        
        u_hat = vec / distance
        # Cleaned up: Assumes both cores are symmetric sizes (self.r + self.r)
        gap = distance - (2 * self.r) 
        
        return u_hat, distance, gap

def run_pure_kinematic_engine(steps=2200, dt=0.005):
    # Initialize cores with a slight spatial offset to allow orbital capture
    core_A = PureKinematicCore2D("Core_A", position=[-4.5, 0.25], velocity=[0.02, -0.05], spin_magnitude=2.5)
    core_B = PureKinematicCore2D("Core_B", position=[4.5, -0.25], velocity=[-0.02, 0.05], spin_magnitude=2.5)
    
    # Bedrock Substrate Constraints (Level N-1 Matrix Floor)
    a_max = 80.0             # Ultimate structural acceleration tolerance of sub-particles
    rho_ambient = 1.0        # Baseline unaligned background density
    v_substrate_sq = 1.0     # The base kinetic energy factor (v^2) of the sub-fluid
    
    # Shared dynamic state of the intervening sub-fluid gap
    rho_gap = 1.0
    
    # Telemetry data storage
    history = {"pos_A": [], "pos_B": [], "distance": [], "rho_gap": [], "P_in": []}
    
    for step in range(steps):
        u_hat, distance, gap = get_interaction_geometry(core_A, core_B.pos)
        effective_gap = max(0.02, gap)
        
        # ---------------------------------------------------------------------
        # STEP 1: CALCULATE CHOKED MASS STAGNATION (The Sub-Scale Ceiling)
        # ---------------------------------------------------------------------
        v_relative = core_B.vel - core_A.vel
        v_close = -np.dot(v_relative, u_hat)
        
        if v_close > 0:
            # Kinematic continuity: shrinking volume demands fluid evacuation speed
            v_escape_required = v_close * (core_A.r / effective_gap)
            a_escape_required = v_escape_required / dt
            
            # Enforce the unyielding structural acceleration limit of the substrate
            a_escape_actual = min(a_escape_required, a_max)
            v_escape_actual = a_escape_actual * dt
            
            # Mass balance ledger: backlog accumulates if volume reduction outpaces exit flow
            d_rho = (v_close * rho_ambient - v_escape_actual * rho_gap) / effective_gap
            rho_gap += d_rho * dt
        else:
            # Relieving compression allows sub-particles to naturally diffuse back to baseline
            rho_gap += (rho_ambient - rho_gap) * 0.1
            
        rho_gap = max(rho_ambient, rho_gap) # Density cannot fall below empty space vacuum floor
        
        # ---------------------------------------------------------------------
        # STEP 2: SOLVE UNIFIED KINEMATIC PRESSURE FIELD: P = ρ * v^2 * (1 - α)
        # ---------------------------------------------------------------------
        # Outer Face: Pure chaotic baseline bombardment (Alignment α = 0)
        P_out = rho_ambient * v_substrate_sq * (1.0 - 0.0)
        
        # Inner Face: Dynamic transition between Macro-Envelope and Micro-Gap
        if gap > 0.4:
            # Mode A: Long-Range Macro-Envelope (Coherent laminar alignment lowers inner pressure)
            alpha_in = 0.85 * np.exp(-0.15 * gap)
        else:
            # Mode B: Short-Range Micro-Gap (Opposing co-rotating rims force a head-on vector clash)
            alpha_in = -1.0
            
        P_in = rho_gap * v_substrate_sq * (1.0 - alpha_in)
        
        # ---------------------------------------------------------------------
        # STEP 3: DERIVE TRANSLATIONAL ACCELERATION FROM PRESSURE GRADIENT
        # ---------------------------------------------------------------------
        # Net vector push acts strictly along the spatial gradient layout (F = -∇P)
        F_gradient = (P_out - P_in) * u_hat * 1.8
        
        # Gyroscopic Torque: Boundary interaction axis crossed with out-of-plane spin vector
        # Resolves in 2D to an orthogonal lateral deflection vector
        u_perpend = np.array([-u_hat[1], u_hat[0]])
        F_gyro_A = u_perpend * (core_A.spin / max(0.5, distance)**2)
        F_gyro_B = -u_perpend * (core_B.spin / max(0.5, distance)**2)
        
        # Sum forces and apply velocity updates simultaneously
        core_A.vel += ((F_gradient + F_gyro_A) / core_A.mass) * dt
        core_B.vel += ((-F_gradient + F_gyro_B) / core_B.mass) * dt
        
        # Translate Coordinates
        core_A.pos += core_A.vel * dt
        core_B.pos += core_B.vel * dt
        
        # Telemetry logging
        history["pos_A"].append(core_A.pos.copy())
        history["pos_B"].append(core_B.pos.copy())
        history["distance"].append(distance)
        history["rho_gap"].append(rho_gap)
        history["P_in"].append(P_in)
        
    return history

# --- EXECUTE AND RENDER DASHBOARD ---
if __name__ == "__main__":
    print("Running Un-Patched Kinematic Pressure Engine...")
    sim_data = run_pure_kinematic_engine()
    
    pos_A = np.array(sim_data["pos_A"])
    pos_B = np.array(sim_data["pos_B"])
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), facecolor='#111116')
    
    # Subplot 1: Pure 2D Trajectory Tracks
    axs[0].set_facecolor('#111116')
    axs[0].plot(pos_A[:, 0], pos_A[:, 1], color='#4d4dff', linewidth=2.5, label='Core A Trajectory')
    axs[0].plot(pos_B[:, 0], pos_B[:, 1], color='#ff4d4d', linewidth=2.5, label='Core B Trajectory')
    axs[0].scatter(pos_A[0, 0], pos_A[0, 1], color='#4d4dff', s=100, label='Start')
    axs[0].scatter(pos_B[0, 0], pos_B[0, 1], color='#ff4d4d', s=100)
    axs[0].scatter(pos_A[-1, 0], pos_A[-1, 1], color='lime', marker='X', s=150, label='Stable Orbit Node')
    axs[0].scatter(pos_B[-1, 0], pos_B[-1, 1], color='lime', marker='X', s=150)
    axs[0].set_title("Emergent Orbital Precession via Pure Pressure Gradient", color='white', fontsize=12, fontweight='bold')
    axs[0].set_xlabel("X Substrate Grid", color='white'); axs[0].set_ylabel("Y Substrate Grid", color='white')
    axs[0].tick_params(colors='white'); axs[0].axis('equal')
    axs[0].grid(color='#222233', linestyle=':', linewidth=0.5)
    axs[0].legend(facecolor='#222233', edgecolor='white', labelcolor='white')
    
    # Subplot 2: Dynamic Substrate Ledger
    axs[1].set_facecolor('#111116')
    steps_axis = np.arange(len(sim_data["distance"]))
    axs[1].plot(steps_axis, sim_data["distance"], color='cyan', linewidth=2.5, label='Separation ($R_{AB}$)')
    axs[1].plot(steps_axis, sim_data["rho_gap"], color='orange', linestyle='--', linewidth=2, label='Gap Density Backlog ($\\rho_{gap}$)')
    axs[1].plot(steps_axis, sim_data["P_in"], color='magenta', linestyle=':', linewidth=2, label='Inner Facing Pressure ($P_{in}$)')
    axs[1].axhline(3.0, color='white', linestyle=':', alpha=0.5, label='Physical Interface Touch (3.0)')
    axs[1].set_title("Dynamic Field Ledger: The Pull & The Stop", color='white', fontsize=12, fontweight='bold')
    axs[1].set_xlabel("Simulation Increments ($dt$)", color='white')
    axs[1].set_ylabel("Metric Magnitudes", color='white')
    axs[1].tick_params(colors='white')
    axs[1].grid(color='#222233', linestyle=':', linewidth=0.5)
    axs[1].legend(facecolor='#222233', edgecolor='white', labelcolor='white')
    
    plt.tight_layout()
    plt.show()