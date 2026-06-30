import numpy as np
import matplotlib.pyplot as plt

class VortexSystem:
    def __init__(self, name, position, initial_velocity):
        self.name = name
        self.x = position
        self.vx = initial_velocity  # Spatial velocity along the interaction axis
        
        # Bedrock Primitives (Isolated resting state)
        self.v_tan_core = 1.0
        self.v_rad_core = 1.0
        self.a = 1.0
        self.zeta = 1.0             # 1.0 = Upright Proton
        
        # Self-solve initial core radius using primitive equation
        self.r = (self.v_tan_core**2 + self.v_rad_core**2) / self.a

    def propagate_field_to_interface(self, r_interface):
        if r_interface <= 0:
            return self.v_tan_core, self.v_rad_core
        # Wave field attenuation profile relative to the core size
        scaling_factor = self.r / (r_interface + self.r)
        return self.v_tan_core * scaling_factor, self.v_rad_core * scaling_factor

# --- The Master System Loop Function ---
def run_dynamic_interaction_solver(steps=500, initial_separation=0.6):
    # Initialize both protons moving toward each other in space
    #system_A = VortexSystem("Proton_A", position=0.0, initial_velocity=0.002)
    #system_B = VortexSystem("Proton_B", position=initial_separation, initial_velocity=-0.002)
    system_A = VortexSystem("Proton_A", position=0.0, initial_velocity=0.005) # Boosted speed
    system_B = VortexSystem("Proton_B", position=0.42, initial_velocity=-0.005) # Closer start
    
    # Substrate Constraint
    r_min_floor = 0.25  # The granular width limit of a single Level N-1 vacuum node
    
    # Tracking logs for relational system_AB tracking
    history = {"distance_AB": [], "zeta_A": [], "zeta_B": [], "a_interface": [], "radius_B": []}
    
    for step in range(steps):
        # 1. Relational Distance (system_AB spatial footprint)
        distance_AB = abs(system_B.x - system_A.x)
        r_interface = distance_AB / 2.0
        
        # 2. Propagate fields from both cores to the 50/50 interface midpoint
        v_tan_A_int, v_rad_A_int = system_A.propagate_field_to_interface(r_interface)
        v_tan_B_int, v_rad_B_int = system_B.propagate_field_to_interface(r_interface)
        
        # 3. Vector Interaction Velocity & Pulse Clash at the interface boundary
        v_interaction_sq = (v_tan_A_int**2 + v_tan_B_int**2 + 
                            2.0 * (v_tan_A_int * v_tan_B_int) * (system_A.zeta * system_B.zeta))
        pulse_clash = (v_rad_A_int * v_rad_B_int) * (system_A.zeta * system_B.zeta)
        
        # 4. Emergent Substrate Impedance (a)
        a_interface = 1.0 + (v_interaction_sq * 1.5) + (pulse_clash * 12.0)
        
        # 5. Symmetrical Workload Handoff: Both particles feel torque and roll axes
        #torque_A = pulse_clash * 0.08
        #torque_B = pulse_clash * 0.08
        torque_A = pulse_clash * 0.45  # Significantly increased from 0.08
        torque_B = pulse_clash * 0.45  # Significantly increased from 0.08
        system_A.zeta = np.clip(system_A.zeta - torque_A, 0.0, 1.0)
        system_B.zeta = np.clip(system_B.zeta - torque_B, 0.0, 1.0)
        
        
        # 6. Spatial Kinematics: The Pressure Gradient Drives Acceleration
        # Head-on pulse clashing pushes them apart; perimeter co-shearing draws them together
        spatial_force = (pulse_clash * 0.04) - (0.0015 / (distance_AB**2))
        
        # Update velocities and positions based on the interface state
        system_A.vx -= spatial_force * 0.1
        system_B.vx += spatial_force * 0.1
        system_A.x += system_A.vx
        system_B.x += system_B.vx
        
        # 7. Apply Primitive Constraint with Node Floor enforcement
        system_A.v_rad_core = system_A.zeta
        system_A.a = a_interface
        system_A.r = max(r_min_floor, (system_A.v_tan_core**2 + system_A.v_rad_core**2) / system_A.a)
        
        system_B.v_rad_core = system_B.zeta
        system_B.a = a_interface
        system_B.r = max(r_min_floor, (system_B.v_tan_core**2 + system_B.v_rad_core**2) / system_B.a)
        
        # Log relational history
        history["distance_AB"].append(distance_AB)
        history["zeta_A"].append(system_A.zeta)
        history["zeta_B"].append(system_B.zeta)
        history["a_interface"].append(a_interface)
        history["radius_B"].append(system_B.r)
        
    return history

# Run Simulation
sim_data = run_dynamic_interaction_solver()

# --- Relational Dashboard Visualization ---
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Top Plot: Global system_AB Separation Distance
axs[0].plot(sim_data["distance_AB"], color='green', linewidth=2.5, label='Spatial Separation ($R_{AB}$)')
axs[0].set_ylabel("Relational Distance", fontsize=11, fontweight='bold')
axs[0].set_title("3D Nucleon Field Solver: Relational system_AB Activation", fontsize=13, fontweight='bold', pad=15)
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].legend(loc='upper right')

# Middle Plot: Symmetrical Coupling Constants
axs[1].plot(sim_data["zeta_A"], color='blue', linewidth=2, label='Zeta A (Particle A)')
axs[1].plot(sim_data["zeta_B"], color='orange', linestyle='--', linewidth=2, label='Zeta B (Particle B)')
axs[1].set_ylabel("Coupling ($\zeta$)", fontsize=11, fontweight='bold')
axs[1].grid(True, linestyle='--', alpha=0.5)
axs[1].legend(loc='upper right')

# Bottom Plot: Core Radius with Substrate Floor Enforcement
axs[2].plot(sim_data["radius_B"], color='purple', linewidth=2, label='Core Radius (Protected Floor = 0.25)')
axs[2].set_ylabel("Core Radius ($r$)", fontsize=11, fontweight='bold')
axs[2].set_xlabel("Simulation Steps (Time)", fontsize=11, fontweight='bold')
axs[2].grid(True, linestyle='--', alpha=0.5)
axs[2].legend(loc='upper right')

plt.tight_layout()
plt.show()