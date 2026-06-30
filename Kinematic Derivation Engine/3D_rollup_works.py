import numpy as np
import matplotlib.pyplot as plt

class PureFluidCore3D:
    def __init__(self, name, position, velocity, spin_axis, core_radius=2.0, c_ceiling=1.2):
        self.name = name
        self.pos = np.array(position, dtype=float)       # [x, y, z]
        self.vel = np.array(velocity, dtype=float)       # [vx, vy, vz]
        
        # Normalize the 3D Spin Axis Vector (Existential Rotation Direction)
        self.spin_axis = np.array(spin_axis, dtype=float)
        self.spin_axis /= np.linalg.norm(self.spin_axis)
        
        self.r = core_radius
        self.c = c_ceiling
        self.v_tan = 1.0
        self.v_rad = 1.0

    def get_boundary_unit_vector(self, partner_pos):
        """Calculates the 3D normalized vector pointing from self to partner"""
        vec = partner_pos - self.pos
        dist = np.linalg.norm(vec)
        if dist == 0:
            return np.array([1.0, 0.0, 0.0]), 0.0
        return vec / dist, dist

def calculate_field_deficit_3d(source_pos, target_pos, K=0.08):
    """Calculates velocity potential deficit at a given 3D coordinate point"""
    R = np.linalg.norm(target_pos - source_pos)
    return K / max(0.1, R)

def run_3d_resonance_engine(steps=2000):
    # Initialize cores with a slight Z-axis offset and unique 3D spin axes
    # Tilting the axes slightly away from pure [0,0,1] creates true 3D precession tracks
    core_A = PureFluidCore3D("Core_A", position=[-4.0, 0.3, 0.1], velocity=[0.005, -0.012, 0.0], spin_axis=[0.0, 0.2, 0.98])
    core_B = PureFluidCore3D("Core_B", position=[4.0, -0.3, -0.1], velocity=[-0.005, 0.012, 0.0], spin_axis=[0.1, -0.1, 0.99])
    
    P_ambient = 1.0
    # Replace: lambda_wave = 4.0
    # Define wavelength dynamically as the sum of the interacting core perimeters
    lambda_wave = core_A.r + core_B.r
    r_node = 0.5
    
    history = {"pos_A": [], "pos_B": [], "distance": [], "impedance": []}
    
    for step in range(steps):
        u_hat, distance = core_A.get_boundary_unit_vector(core_B.pos)
        r_interface = distance / 2.0
        
        # 1. Map 3D Inner and Outer boundary edge coordinates
        x_out_A = core_A.pos - (u_hat * core_A.r)
        x_in_A = core_A.pos + (u_hat * core_A.r)
        x_in_B = core_B.pos - (u_hat * core_B.r)
        x_out_B = core_B.pos + (u_hat * core_B.r)
        
        # 2. Solve 3D edge pressures via field deficits
        P_out_A = P_ambient - calculate_field_deficit_3d(core_B.pos, x_out_A)
        P_in_A = P_ambient - calculate_field_deficit_3d(core_B.pos, x_in_A)
        P_out_B = P_ambient - calculate_field_deficit_3d(core_A.pos, x_out_B)
        P_in_B = P_ambient - calculate_field_deficit_3d(core_A.pos, x_in_B)
        
        # 3. Long-range linear gravitational push along the line of centers
        accel_A = (P_out_A - P_in_A) * u_hat * 0.1
        accel_B = (P_out_B - P_in_B) * (-u_hat) * 0.1
        core_A.vel += accel_A
        core_B.vel += accel_B
        
        # 4. THE 3D GYROSCOPIC PIVOT MECHANIC
        # Near-field compression torque forces an orthogonal deflection vector 
        if distance <= (core_A.r + core_B.r) * 1.5:
            torque_scale = 0.015 / max(0.1, distance)**2
            
            # Cross product of the interaction axis and internal spin vector dictates the dodge vector
            gyro_dodge_A = np.cross(u_hat, core_A.spin_axis) * torque_scale
            gyro_dodge_B = np.cross(-u_hat, core_B.spin_axis) * torque_scale
            
            core_A.vel += gyro_dodge_A
            core_B.vel += gyro_dodge_B

        # 5. Continuous Wave-Phase Resonance Impedance Filter (Uniform Drag)
        resonance_factor = np.sin(np.pi * distance / lambda_wave)**2
        dilution = (r_node / max(r_node, r_interface))**3
        impedance = resonance_factor * dilution * 0.06
        
        # Tax all velocity components identically
        core_A.vel -= core_A.vel * impedance
        core_B.vel -= core_B.vel * impedance
        
        # 6. Global v=c saturation velocity ceiling budget
        for core in [core_A, core_B]:
            v_total = np.sqrt(core.v_tan**2 + core.v_rad**2 + np.sum(core.vel**2))
            if v_total > core.c:
                core.vel *= (core.c / v_total)

        # 7. Translate 3D Coordinates
        core_A.pos += core_A.vel
        core_B.pos += core_B.vel
        
        # Telemetry logging
        history["pos_A"].append(core_A.pos.copy())
        history["pos_B"].append(core_B.pos.copy())
        history["distance"].append(distance)
        history["impedance"].append(impedance)
        
    return history

# --- Render 3D Dashboard ---
sim_data = history_3d = run_resonance_filter_engine_3d = run_3d_resonance_engine()
pos_A = np.array(sim_data["pos_A"])
pos_B = np.array(sim_data["pos_B"])

fig = plt.figure(figsize=(14, 6))

# Subplot 1: True 3D Trajectory Track Space
axs0 = fig.add_subplot(121, projection='3d')
axs0.plot(pos_A[:, 0], pos_A[:, 1], pos_A[:, 2], color='blue', linewidth=2, label='Core A 3D Track')
axs0.plot(pos_B[:, 0], pos_B[:, 1], pos_B[:, 2], color='red', linewidth=2, label='Core B 3D Track')
axs0.scatter(pos_A[0, 0], pos_A[0, 1], pos_A[0, 2], color='blue', s=80)
axs0.scatter(pos_B[0, 0], pos_B[0, 1], pos_B[0, 2], color='red', s=80)
axs0.scatter(pos_A[-1, 0], pos_A[-1, 1], pos_A[-1, 2], color='purple', marker='X', s=120, label='Settle Position')
axs0.scatter(pos_B[-1, 0], pos_B[-1, 1], pos_B[-1, 2], color='purple', marker='X', s=120)
axs0.set_title("3D Space Paths: Gyroscopic Precession Deflection", fontsize=11, fontweight='bold')
axs0.set_xlabel("X Axis")
axs0.set_ylabel("Y Axis")
axs0.set_zlabel("Z Axis")
axs0.legend()

# Subplot 2: 3D Separation Tracking
axs1 = fig.add_subplot(122)
axs1.plot(sim_data["distance"], color='green', linewidth=2.5, label='3D Separation ($R_{AB}$)')
axs1.axhline(4.0, color='black', linestyle=':', label='Harmonic Node (4.0)')
axs1.set_title("3D Separation Profile Over Time", fontsize=11, fontweight='bold')
axs1.set_xlabel("Simulation Steps", fontweight='bold')
axs1.set_ylabel("Distance Magnitude", fontweight='bold')
axs1.grid(True, linestyle='--', alpha=0.5)
axs1.legend()

plt.tight_layout()
plt.show()