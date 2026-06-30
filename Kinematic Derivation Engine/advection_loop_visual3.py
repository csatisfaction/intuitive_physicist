import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ProxyTrackedToroid:
    def __init__(self, center, major_radius, minor_radius, vorticity_vec, starting_vel=[0.0, 0.0, 0.0], c=1.0):
        self.center = np.array(center, dtype=np.float64)
        self.R = float(major_radius)
        self.r_m = float(minor_radius)
        self.omega = np.array(vorticity_vec, dtype=np.float64)
        self.vel = np.array(starting_vel, dtype=np.float64)
        self.c = float(c)
        
        self.omega_mag = np.linalg.norm(self.omega)
        self.circulation = self.omega_mag * np.pi * (self.r_m ** 2)
        self.update_basis()

    def update_basis(self):
        self.omega_mag = np.linalg.norm(self.omega)
        if self.omega_mag > 1e-12:
            self.normal = self.omega / self.omega_mag
        else:
            self.normal = np.array([0.0, 0.0, 1.0])
        
        # Orthonormal basis for ring configuration in 3D space
        arbitrary_vec = np.array([0.0, 0.0, 1.0]) if abs(self.normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        self.u_basis = np.cross(self.normal, arbitrary_vec)
        self.u_basis /= np.linalg.norm(self.u_basis)
        self.v_basis = np.cross(self.normal, self.u_basis)

    def get_filament_nodes(self, num_segments=32):
        theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        cos_t = np.cos(theta)[:, np.newaxis]
        sin_t = np.sin(theta)[:, np.newaxis]
        return self.center + self.R * (cos_t * self.u_basis + sin_t * self.v_basis)

    def project_velocity_field(self, target_point, num_segments=32):
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

def step_true_tir_system(pA, pB, dt, G_stiffness=0.15):
    """
    Simulates a time step using the Bernoulli Pressure Gradient proxy for spatial 
    translation alongside pure emergent gyroscopic torques for axial tilt.
    """
    # 1. CALCULATE PURE EMERGENT GYROSCOPIC AXIAL TORQUES (Node-level sampling)
    nodes_A = pA.get_filament_nodes()
    v_ext_at_A = np.array([pB.project_velocity_field(n) for n in nodes_A])
    torque_A = np.mean(np.cross(nodes_A - pA.center, v_ext_at_A), axis=0)
    
    nodes_B = pB.get_filament_nodes()
    v_ext_at_B = np.array([pA.project_velocity_field(n) for n in nodes_B])
    torque_B = np.mean(np.cross(nodes_B - pB.center, v_ext_at_B), axis=0)
    
    # 2. CALCULATE CO-SHEAR BERNOULLI PRESSURE GRADIENTS (Proxy shortcut for attraction)
    eps = 0.02
    grad_v_sq_A = np.zeros(3)
    grad_v_sq_B = np.zeros(3)
    
    for axis in range(3):
        nudge = np.zeros(3)
        nudge[axis] = eps
        
        # Probe field around Particle B induced by Particle A
        v_plus_B = np.linalg.norm(pA.project_velocity_field(pB.center + nudge))
        v_minus_B = np.linalg.norm(pA.project_velocity_field(pB.center - nudge))
        grad_v_sq_B[axis] = (v_plus_B**2 - v_minus_B**2) / (2 * eps)
        
        # Probe field around Particle A induced by Particle B
        v_plus_A = np.linalg.norm(pB.project_velocity_field(pA.center + nudge))
        v_minus_A = np.linalg.norm(pB.project_velocity_field(pA.center - nudge))
        grad_v_sq_A[axis] = (v_plus_A**2 - v_minus_A**2) / (2 * eps)

    # 3. APPLY KINEMATIC TRANSFORMATIONS
    # Pressure drop accelerates cores toward zones of maximum localized field velocity
    pA.vel += grad_v_sq_A * G_stiffness * dt
    pB.vel += grad_v_sq_B * G_stiffness * dt
    
    # Position updates via momentum velocities
    pA.center += pA.vel * dt
    pB.center += pB.vel * dt
    
    # Precession updates via localized ring torques
    pA.omega += torque_A * dt
    pB.omega += torque_B * dt
    
    pA.update_basis()
    pB.update_with_basis_normalization() if hasattr(pB, 'update_with_basis_normalization') else pB.update_basis()

# --- RUNTIMELINE SIMULATION ENGINE ---
if __name__ == "__main__":
    # Particle A: Stationary high-torque proton anchor centered at the origin
    proton = ProxyTrackedToroid(
        center=[0.0, 0.0, 0.0], 
        major_radius=0.5, 
        minor_radius=0.1, 
        vorticity_vec=[0.0, 0.0, 45.0], # Fast vertical rotation
        starting_vel=[0.0, 0.0, 0.0]
    )
    
    # Particle B: Lighter test core offset at X=2.2.
    # Initialized with a Y-directed tangential starting velocity so it falls *around* the proton
    # Tilted intentionally into the X-Y plane to observe gyroscopic gear alignment
    test_core = ProxyTrackedToroid(
        center=[2.2, 0.0, 0.0], 
        major_radius=0.4, 
        minor_radius=0.1, 
        vorticity_vec=[12.0, 12.0, 0.0], 
        starting_vel=[0.0, 0.28, 0.0] # Pristine perpendicular starting vector
    )
    
    steps = 600
    dt = 0.04
    
    # Registers for tracking history arrays
    posA_hist, posB_hist = np.zeros((steps, 3)), np.zeros((steps, 3))
    normA_hist, normB_hist = np.zeros((steps, 3)), np.zeros((steps, 3))
    
    print("=== EXECUTING SYSTEM ENGINE WITH BERNOULLI PROXIES ===")
    for t in range(steps):
        posA_hist[t] = proton.center.copy()
        posB_hist[t] = test_core.center.copy()
        normA_hist[t] = proton.normal.copy()
        normB_hist[t] = test_core.normal.copy()
        
        step_true_tir_system(proton, test_core, dt)
        
    print(f"Simulation completed across {steps} execution segments.")
    
    # --- MATPLOTLIB VISUALIZATION ---
    fig = plt.figure(figsize=(15, 7), facecolor='#111116')
    
    # PANEL 1: True Curved Trajectories
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor('#111116')
    ax1.xaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    ax1.yaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    ax1.zaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    
    ax1.plot(posA_hist[:,0], posA_hist[:,1], posA_hist[:,2], color='magenta', linewidth=2.0, label='Proton Anchor')
    ax1.plot(posB_hist[:,0], posB_hist[:,1], posB_hist[:,2], color='cyan', linewidth=2.0, label='Orbital Core (Proxy Well)')
    ax1.scatter(posA_hist[-1,0], posA_hist[-1,1], posA_hist[-1,2], color='magenta', s=40)
    ax1.scatter(posB_hist[-1,0], posB_hist[-1,1], posB_hist[-1,2], color='cyan', s=40)
    
    ax1.set_title("Emergent Substrate Trajectories (True Orbital Path)", color='white', fontsize=12, pad=10)
    ax1.set_xlabel("X Space", color='white'); ax1.set_ylabel("Y Space", color='white'); ax1.set_zlabel("Z Space", color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#1c1c24', edgecolor='none', labelcolor='white')
    
    # PANEL 2: Clean Precessional Cone Spherical Mapping
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor('#111116')
    ax2.xaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    ax2.yaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    ax2.zaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    
    # Fixed wireframe sphere construction
    u_grid, v_grid = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xs = np.cos(u_grid) * np.sin(v_grid)
    ys = np.sin(u_grid) * np.sin(v_grid)
    zs = np.cos(v_grid)
    ax2.plot_wireframe(xs, ys, zs, color='#222233', linewidth=0.5, alpha=0.4)
    
    ax2.plot(normA_hist[:,0], normA_hist[:,1], normA_hist[:,2], color='magenta', linewidth=1.5, label='Anchor Normal')
    ax2.plot(normB_hist[:,0], normB_hist[:,1], normB_hist[:,2], color='cyan', linewidth=2.5, label='Orbital Axis Precession')
    ax2.scatter(normB_hist[0,0], normB_hist[0,1], normB_hist[0,2], color='yellow', s=60, marker='*', label='Precession Start')
    ax2.scatter(normB_hist[-1,0], normB_hist[-1,1], normB_hist[-1,2], color='cyan', s=30)
    
    ax2.set_title("Spin Orientation Track (Pristine Gyroscopic Precession)", color='white', fontsize=12, pad=10)
    ax2.set_xlabel("Nx Axis", color='white'); ax2.set_ylabel("Ny Axis", color='white'); ax2.set_zlabel("Nz Axis", color='white')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#1c1c24', edgecolor='none', labelcolor='white')
    
    ax2.set_xlim([-1, 1]); ax2.set_ylim([-1, 1]); ax2.set_zlim([-1, 1])
    
    plt.tight_layout()
    plt.show()