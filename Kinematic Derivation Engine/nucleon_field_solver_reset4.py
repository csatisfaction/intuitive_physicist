import numpy as np
import matplotlib.pyplot as plt

class VortexSystem:
    def __init__(self, name, position, initial_velocity):
        self.name = name
        self.x = position
        self.vx = initial_velocity  
        
        # Bedrock Primitives (Isolated resting state)
        self.v_tan_core = 1.0
        self.v_rad_core = 1.0
        self.a = 1.0
        self.zeta = 1.0             
        
        # Self-solve initial core radius using primitive equation
        self.r = (self.v_tan_core**2 + self.v_rad_core**2) / self.a

    def propagate_field_to_interface(self, r_interface):
        if r_interface <= 0:
            return self.v_tan_core, self.v_rad_core
        scaling_factor = self.r / (r_interface + self.r)
        return self.v_tan_core * scaling_factor, self.v_rad_core * scaling_factor

def run_isolated_experiment(steps=1000, initial_separation=0.6):
    # Keep the original slow, heavy macro velocities
    system_A = VortexSystem("Proton_A", position=0.0, initial_velocity=0.002)
    system_B = VortexSystem("Proton_B", position=initial_separation, initial_velocity=-0.002)
    
    r_min_floor = 0.25  
    history = {"distance_AB": [], "zeta_B": [], "radius_B": [], "a_interface": []}
    
    for step in range(steps):
        # 1. Relational Distance (system_AB)
        distance_AB = abs(system_B.x - system_A.x)
        r_interface = distance_AB / 2.0
        
        # 2. Propagate to 50/50 interface
        v_tan_A_int, v_rad_A_int = system_A.propagate_field_to_interface(r_interface)
        v_tan_B_int, v_rad_B_int = system_B.propagate_field_to_interface(r_interface)
        
        # 3. Pure Vector Interface Math
        v_interaction_sq = (v_tan_A_int**2 + v_tan_B_int**2 + 
                            2.0 * (v_tan_A_int * v_tan_B_int) * (system_A.zeta * system_B.zeta))
        pulse_clash = (v_rad_A_int * v_rad_B_int) * (system_A.zeta * system_B.zeta)
        
        # 4. Emergent Substrate Impedance (a)
        a_interface = 1.0 + (v_interaction_sq * 1.5) + (pulse_clash * 12.0)
        
        # 5. Pure Primitives Torque (Original unboosted coefficients)
        torque_A = pulse_clash * 0.08
        torque_B = pulse_clash * 0.08
        system_A.zeta = np.clip(system_A.zeta - torque_A, 0.0, 1.0)
        system_B.zeta = np.clip(system_B.zeta - torque_B, 0.0, 1.0)
        
        # 6. Spatial Kinematics (Original unboosted forces, NO artificial drag)
        spatial_force = (pulse_clash * 0.04) - (0.0015 / (distance_AB**2))
        
        system_A.vx -= spatial_force * 0.1
        system_B.vx += spatial_force * 0.1
        system_A.x += system_A.vx
        system_B.x += system_B.vx
        
        # 7. Apply Primitive Constraint
        system_A.v_rad_core = system_A.zeta
        system_A.a = a_interface
        system_A.r = max(r_min_floor, (system_A.v_tan_core**2 + system_A.v_rad_core**2) / system_A.a)
        
        system_B.v_rad_core = system_B.zeta
        system_B.a = a_interface
        system_B.r = max(r_min_floor, (system_B.v_tan_core**2 + system_B.v_rad_core**2) / system_B.a)
        
        history["distance_AB"].append(distance_AB)
        history["zeta_B"].append(system_B.zeta)
        history["radius_B"].append(system_B.r)
        history["a_interface"].append(a_interface)
        
    return history

# Run and Plot
sim_data = run_isolated_experiment()
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(sim_data["distance_AB"], color='green', linewidth=2, label='Spatial Separation ($R_{AB}$)')
axs[0].set_ylabel("Relational Distance", fontweight='bold')
axs[0].set_title("Pure Isolated Parameter Experiment (1000 Steps)", fontsize=12, fontweight='bold')
axs[0].grid(True, linestyle='--')
axs[0].legend()

axs[1].plot(sim_data["zeta_B"], color='orange', linewidth=2, label='Zeta B (Tilt)')
axs[1].set_ylabel("Coupling ($\zeta$)", fontweight='bold')
axs[1].grid(True, linestyle='--')
axs[1].legend()

axs[2].plot(sim_data["a_interface"], color='red', linewidth=2, label='Impedance ($a$)')
axs[2].set_ylabel("Substrate 'a'", fontweight='bold')
axs[2].set_xlabel("Steps")
axs[2].grid(True, linestyle='--')
axs[2].legend()

plt.tight_layout()
plt.show()