import numpy as np

class InteractiveToroid:
    def __init__(self, center, major_radius, minor_radius, vorticity_vec, c=1.0):
        self.center = np.array(center, dtype=np.float64)
        self.R = float(major_radius)
        self.r_m = float(minor_radius)
        self.omega = np.array(vorticity_vec, dtype=np.float64)
        self.c = float(c)
        
        # Calculate circulation strength
        self.omega_mag = np.linalg.norm(self.omega)
        self.circulation = self.omega_mag * np.pi * (self.r_m ** 2)
        self.update_basis()

    def update_basis(self):
        """Re-calculates the spatial orientation plane based on current vorticity vector."""
        self.omega_mag = np.linalg.norm(self.omega)
        self.normal = self.omega / self.omega_mag
        
        arbitrary_vec = np.array([0.0, 0.0, 1.0]) if abs(self.normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        self.u_basis = np.cross(self.normal, arbitrary_vec)
        self.u_basis /= np.linalg.norm(self.u_basis)
        self.v_basis = np.cross(self.normal, self.u_basis)

    def get_filament_nodes(self, num_segments=32):
        """Generates the 3D coordinates of the discrete nodes forming the major radius ring."""
        theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        cos_t = np.cos(theta)[:, np.newaxis]
        sin_t = np.sin(theta)[:, np.newaxis]
        return self.center + self.R * (cos_t * self.u_basis + sin_t * self.v_basis)

    def calculate_induced_velocity(self, target_point, num_segments=32):
        """Calculates the velocity vector this specific particle projects into a single coordinate."""
        x = np.array(target_point, dtype=np.float64)
        v_raw = np.zeros(3, dtype=np.float64)
        ring_nodes = self.get_filament_nodes(num_segments)
        
        for i in range(num_segments):
            node_current = ring_nodes[i]
            node_next = ring_nodes[(i + 1) % num_segments]
            
            delta_s = node_next - node_current
            midpoint = (node_current + node_next) * 0.5
            r_vec = x - midpoint
            r_mag_sq = np.dot(r_vec, r_vec)
            
            denominator = (r_mag_sq + (self.r_m ** 2)) ** 1.5
            cross_prod = np.cross(delta_s, r_vec)
            
            v_raw += (self.circulation / (4 * np.pi)) * (cross_prod / denominator)
            
        v_mag = np.linalg.norm(v_raw)
        if v_mag > self.c:
            return self.c * (v_raw / v_mag)
        return v_raw

def step_two_body_system(pA, pB, dt):
    """
    Executes a single, pure Lagrangian back-reaction step.
    Each particle samples the field gradient across its own diameter to calculate drift and tilt.
    """
    # 1. Sample the field of Particle B at all of Particle A's filament coordinates
    nodes_A = pA.get_filament_nodes()
    v_ext_at_A = np.array([pB.calculate_induced_velocity(n) for n in nodes_A])
    
    # 2. Sample the field of Particle A at all of Particle B's filament coordinates
    nodes_B = pB.get_filament_nodes()
    v_ext_at_B = np.array([pA.calculate_induced_velocity(n) for n in nodes_B])
    
    # --- UPDATE PARTICLE A KINEMATICS ---
    # Linear drift (Average background flow)
    drift_A = np.mean(v_ext_at_A, axis=0)
    # Gyroscopic Torque: Sum of (r x v_external)
    levers_A = nodes_A - pA.center
    torques_A = np.cross(levers_A, v_ext_at_A)
    net_torque_A = np.mean(torques_A, axis=0)
    
    # --- UPDATE PARTICLE B KINEMATICS ---
    drift_B = np.mean(v_ext_at_B, axis=0)
    levers_B = nodes_B - pB.center
    torques_B = np.cross(levers_B, v_ext_at_B)
    net_torque_B = np.mean(torques_B, axis=0)
    
    # Apply differential step simultaneously to keep interaction pure
    pA.center += drift_A * dt
    pA.omega  += net_torque_A * dt
    pA.update_basis()
    
    pB.center += drift_B * dt
    pB.omega  += net_torque_B * dt
    pB.update_basis()

# Sandbox Execution: Let's run a test timeline
if __name__ == "__main__":
    # Particle A: Proton-like core resting at the origin, spinning vertically
    proton = InteractiveToroid(center=[0.0, 0.0, 0.0], major_radius=0.5, minor_radius=0.1, vorticity_vec=[0.0, 0.0, 10.0])
    
    # Particle B: A slightly offset test core nearby, tilted at an angle
    test_core = InteractiveToroid(center=[2.0, 0.0, 0.0], major_radius=0.5, minor_radius=0.1, vorticity_vec=[5.0, 5.0, 0.0])
    
    print("=== STARTING INTERACTIVE BACK-REACTION SANDBOX ===")
    print(f"Initial Proton Center:    {proton.center} | Spin Axis: {proton.normal}")
    print(f"Initial Test Core Center: {test_core.center} | Spin Axis: {test_core.normal}\n")
    
    # Advance time through 3 sequential cycles to observe the feedback
    dt = 0.1
    for step in range(1, 4):
        step_two_body_system(proton, test_core, dt)
        print(f"--- TIMESTEP T = {step*dt:.1f} ---")
        print(f"Proton Center:    {proton.center} | Spin Axis: {proton.normal}")
        print(f"Test Core Center: {test_core.center} | Spin Axis: {test_core.normal}")