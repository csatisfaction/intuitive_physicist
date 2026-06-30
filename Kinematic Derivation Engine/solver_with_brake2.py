import numpy as np
import matplotlib.pyplot as plt

class PureKinematicCore:
    def __init__(self, name, position, initial_velocity):
        self.name = name
        self.x = position
        self.vx = initial_velocity
        
        # Primitives
        self.v_tan = 1.0
        self.v_rad = 1.0
        # Auto-solve radius based on baseline internal containment a=1.0
        self.r = (self.v_tan**2 + self.v_rad**2) / 1.0 # Resolves to 2.0
        
        # Non-linear phase state trackers
        self.is_pinned = False

    def get_edges(self, partner_x):
        """Identifies inner edge (facing partner) and outer edge (facing vacuum)"""
        if self.x < partner_x:
            x_outer = self.x - self.r
            x_inner = self.x + self.r
        else:
            x_outer = self.x + self.r
            x_inner = self.x - self.r
        return x_outer, x_inner

def calculate_field_deficit(source_core, target_x, K=0.015):
    """Calculates neighbor's field deficit dropping off at 1/R"""
    R = abs(target_x - source_core.x)
    R = max(0.05, R) # Prevent singularity clip
    return K / R

def run_harmonic_lock_sim(steps=800, initial_separation=8.0):
    # Initialize cores far apart in unshielded space
    core_A = PureKinematicCore("Core_A", position=0.0, initial_velocity=0.001)
    core_B = PureKinematicCore("Core_B", position=initial_separation, initial_velocity=-0.001)
    
    P_ambient = 1.0
    history = {"distance": [], "P_outer_A": [], "P_inner_A": [], "vx_A": []}
    
    for step in range(steps):
        # 1. Track relational distance between singularity centers
        distance = abs(core_B.x - core_A.x)
        touching_threshold = core_A.r + core_B.r # 2.0 + 2.0 = 4.0 whole integer boundary
        
        # 2. Check for Non-Linear Resonance Lock-In (The Harmonic Filter Rule)
        if not core_A.is_pinned and distance <= touching_threshold:
            core_A.is_pinned = True
            core_B.is_pinned = True
            # Macro translation kinetic energy is fully spent and pinned into the coordinate node
            core_A.vx = 0.0
            core_B.vx = 0.0
            
            # Enforce rigid locking boundary alignment geometry
            # Prevents numeric creep from drifting the pinned state
            midpoint = (core_A.x + core_B.x) / 2.0
            core_A.x = midpoint - (touching_threshold / 2.0)
            core_B.x = midpoint + (touching_threshold / 2.0)

        # 3. Process kinematics only if the field hasn't pinned into a ground state
        if not core_A.is_pinned:
            # Locate active outer/inner interaction planes
            x_out_A, x_in_A = core_A.get_edges(core_B.x)
            x_out_B, x_in_B = core_B.get_edges(core_A.x)
            
            # Solve edge pressures via field deficit summation
            P_out_A = P_ambient - calculate_field_deficit(core_B, x_out_A)
            P_in_A = P_ambient - calculate_field_deficit(core_B, x_in_A)
            P_out_B = P_ambient - calculate_field_deficit(core_A, x_out_B)
            P_in_B = P_ambient - calculate_field_deficit(core_A, x_in_B)
            
            # Resolve relative axis orientation
            direction_sign = np.sign(core_B.x - core_A.x)
            
            # Calculate pressure imbalance magnitude across the flywheels
            accel_A = (P_out_A - P_in_A) * 0.1
            accel_B = (P_out_B - P_in_B) * 0.1
            
            # Displace coordinates smoothly along the pressure gradient
            core_A.vx += accel_A * direction_sign
            core_B.vx -= accel_B * direction_sign
            
            core_A.x += core_A.vx
            core_B.x += core_B.vx
        else:
            # While pinned, pressures and translation remain locked at the node minimum
            P_out_A = P_ambient - calculate_field_deficit(core_B, core_A.x - core_A.r)
            P_in_A = P_ambient - calculate_field_deficit(core_B, core_A.x + core_A.r)
            
        # Log telemetry metrics
        history["distance"].append(distance)
        history["P_outer_A"].append(P_out_A)
        history["P_inner_A"].append(P_in_A)
        history["vx_A"].append(core_A.vx)
        
    return history

# --- Render Simulation Outputs ---
sim_data = run_harmonic_lock_sim()
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Graph 1: Separation Record
axs[0].plot(sim_data["distance"], color='green', linewidth=2.5, label='Macroscopic Gap ($R_{AB}$)')
axs[0].axhline(4.0, color='black', linestyle=':', label='Resonance Boundary ($r_A+r_B$)')
axs[0].set_ylabel("Relational Distance", fontsize=11, fontweight='bold')
axs[0].set_title("Two-Point Pressure Sandbox: Emergent Gravity & Harmonic Pinning Lock", fontsize=13, fontweight='bold', pad=15)
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].legend()

# Graph 2: Macro Velocity Profile
axs[1].plot(sim_data["vx_A"], color='purple', linewidth=2, label='Spatial Velocity ($v_x$)')
axs[1].set_ylabel("Translation $v_x$", fontsize=11, fontweight='bold')
axs[1].grid(True, linestyle='--', alpha=0.5)
axs[1].legend()

# Graph 3: Substrate Ledger Reports
axs[2].plot(sim_data["P_outer_A"], color='blue', linewidth=2, label='Outer Face Pressure')
axs[2].plot(sim_data["P_inner_A"], color='red', linestyle='--', linewidth=2, label='Inner Channel Pressure')
axs[2].set_ylabel("Vacuum Potential ($v^2$)", fontsize=11, fontweight='bold')
axs[2].set_xlabel("Simulation Steps (Time Line)", fontsize=11, fontweight='bold')
axs[2].grid(True, linestyle='--', alpha=0.5)
axs[2].legend()

plt.tight_layout()
plt.show()