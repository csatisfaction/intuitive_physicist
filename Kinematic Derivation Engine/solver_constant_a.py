import numpy as np
import matplotlib.pyplot as plt

class VortexSystem:
    def __init__(self, name, position, initial_velocity):
        self.name = name
        self.x = position
        self.vx = initial_velocity  # Spatial macro-velocity
        
        # Bedrock Primitives (Isolated resting state)
        self.v_tan_core = 1.0
        self.v_rad_core = 1.0
        self.zeta = 1.0             # Rotational alignment vector projection
        
        # Self-solve initial core radius with 'a' locked at baseline 1.0
        self.r = (self.v_tan_core**2 + self.v_rad_core**2) / 1.0

    def propagate_field_to_interface(self, r_interface):
        if r_interface <= 0:
            return self.v_tan_core, self.v_rad_core
        # Volumetric wave field attenuation profile relative to core scale
        scaling_factor = self.r / (r_interface + self.r)
        return self.v_tan_core * scaling_factor, self.v_rad_core * scaling_factor

def collision_dispersion(v_clash, r_interface, r_node=0.01):
    """
    Sub-Scale 3D Dispersal Function (Level N-1 Vacuum Matrix).
    Spreads interface velocity clash out into the volume of background nodes.
    The velocity tax drops off by the cube of the interface radius.
    """
    R = max(r_node, r_interface)
    v_node_dispersed = v_clash * (r_node / R)**3
    return v_node_dispersed

def run_velocity_ledger_solver(steps=600, initial_separation=0.55):
    # Initialize both systems moving toward each other
    system_A = VortexSystem("Proton_A", position=0.0, initial_velocity=0.0015)
    system_B = VortexSystem("Proton_B", position=initial_separation, initial_velocity=-0.0015)
    
    # History logs for dynamic verification
    history = {
        "distance_AB": [],
        "v_rad_core_B": [],
        "radius_B": [],
        "v_tax_leeched": []
    }
    
    for step in range(steps):
        # 1. Relational Footprint (system_AB spatial gap)
        distance_AB = abs(system_B.x - system_A.x)
        r_interface = distance_AB / 2.0
        
        # 2. Field Propagation to the 50/50 interface midpoint
        v_tan_A_int, v_rad_A_int = system_A.propagate_field_to_interface(r_interface)
        v_tan_B_int, v_rad_B_int = system_B.propagate_field_to_interface(r_interface)
        
        # 3. Resolve head-on wave-front stagnation clash
        pulse_clash = (v_rad_A_int * v_rad_B_int) * (system_A.zeta * system_B.zeta)
        
        # 4. Nested Dispersion: Calculate the Substrate Volumetric Tax (1/R^3)
        v_tax = collision_dispersion(v_clash=pulse_clash, r_interface=r_interface, r_node=0.01)
        
        # 5. Pythagorean Vector Subtraction (Deducting tax from core energy budgets)
        system_A.v_rad_core = np.sqrt(max(0.0, system_A.v_rad_core**2 - v_tax**2))
        system_B.v_rad_core = np.sqrt(max(0.0, system_B.v_rad_core**2 - v_tax**2))
        
        # 6. Auto-Solve Radius via the Pure Primitive Constraint
        system_A.r = (system_A.v_tan_core**2 + system_A.v_rad_core**2) / 1.0
        system_B.r = (system_B.v_tan_core**2 + system_B.v_rad_core**2) / 1.0
        
        # 7. Spatial Kinematics driven by the local vector gradient dispersal
        spatial_force = pulse_clash - (0.0012 / (distance_AB**2))
        system_A.vx -= spatial_force * 0.1
        system_B.vx += spatial_force * 0.1
        system_A.x += system_A.vx
        system_B.x += system_B.vx
        
        # Track parameters
        history["distance_AB"].append(distance_AB)
        history["v_rad_core_B"].append(system_B.v_rad_core)
        history["radius_B"].append(system_B.r)
        history["v_tax_leeched"].append(v_tax)
        
    return history

# --- Execute and Generate Output Charts ---
sim_data = run_velocity_ledger_solver()
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Graph 1: Relational Separation (system_AB footprint)
axs[0].plot(sim_data["distance_AB"], color='green', linewidth=2.5, label='Spatial Separation ($R_{AB}$)')
axs[0].set_ylabel("Relational Distance", fontsize=11, fontweight='bold')
axs[0].set_title("Pure Velocity Ledger Solver: 3D Volumetric Dispersal ($1/R^3$)", fontsize=13, fontweight='bold', pad=15)
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].legend(loc='upper right')

# Graph 2: Core Velocity Evolution (v_rad depletion)
axs[1].plot(sim_data["v_rad_core_B"], color='blue', linewidth=2, label='Core Radial Velocity ($v_{rad}$)')
axs[1].plot(sim_data["v_tax_leeched"], color='red', linestyle=':', linewidth=2, label='Substrate Tax ($v_{tax}$)')
axs[1].set_ylabel("Velocity Magnitudes", fontsize=11, fontweight='bold')
axs[1].grid(True, linestyle='--', alpha=0.5)
axs[1].legend(loc='upper right')

# Graph 3: Core Radius Auto-Resolution
axs[2].plot(sim_data["radius_B"], color='purple', linewidth=2, label='Core Radius ($r = v^2 / 1.0$)')
axs[2].set_ylabel("Core Radius ($r$)", fontsize=11, fontweight='bold')
axs[2].set_xlabel("Simulation Steps (Time)", fontsize=11, fontweight='bold')
axs[2].grid(True, linestyle='--', alpha=0.5)
axs[2].legend(loc='upper right')

plt.tight_layout()
plt.show()