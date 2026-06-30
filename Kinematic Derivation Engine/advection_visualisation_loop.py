import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TrackedToroid:
    def __init__(self, center, major_radius, minor_radius, vorticity_vec, c=1.0):
        self.center = np.array(center, dtype=np.float64)
        self.R = float(major_radius)
        self.r_m = float(minor_radius)
        self.omega = np.array(vorticity_vec, dtype=np.float64)
        self.c = float(c)
        
        self.omega_mag = np.linalg.norm(self.omega)
        self.circulation = self.omega_mag * np.pi * (self.r_m ** 2)
        self.update_basis()

    def update_basis(self):
        self.omega_mag = np.linalg.norm(self.omega)
        self.normal = self.omega / self.omega_mag
        
        # Orthonormal basis for ring configuration
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
        
        # Vectorized ring nodes
        theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        cos_t = np.cos(theta)[:, np.newaxis]
        sin_t = np.sin(theta)[:, np.newaxis]
        ring_nodes = self.center + self.R * (cos_t * self.u_basis + sin_t * self.v_basis)
        
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

def run_advection_loop(pA, pB, steps=400, dt=0.01):
    """
    Executes a high-fidelity relational interaction loop, storing full coordinate 
    and orientation histories for structural analysis.
    """
    # Initialize history registers
    hist_pos_A = np.zeros((steps, 3))
    hist_pos_B = np.zeros((steps, 3))
    hist_n_A   = np.zeros((steps, 3))
    hist_n_B   = np.zeros((steps, 3))
    
    for t in range(steps):
        # Record states prior to transformation
        hist_pos_A[t] = pA.center.copy()
        hist_pos_B[t] = pB.center.copy()
        hist_n_A[t]   = pA.normal.copy()
        hist_n_B[t]   = pB.normal.copy()
        
        # 1. Particle A samples Particle B's projected substrate velocities
        nodes_A = pA.get_filament_nodes()
        v_ext_at_A = np.array([pB.project_velocity_field(n) for n in nodes_A])
        
        # 2. Particle B samples Particle A's projected substrate velocities
        nodes_B = pB.get_filament_nodes()
        v_ext_at_B = np.array([pA.project_velocity_field(n) for n in nodes_B])
        
        # Calculate kinematics for A
        drift_A = np.mean(v_ext_at_A, axis=0)
        levers_A = nodes_A - pA.center
        torque_A = np.mean(np.cross(levers_A, v_ext_at_A), axis=0)
        
        # Calculate kinematics for B
        drift_B = np.mean(v_ext_at_B, axis=0)
        levers_B = nodes_B - pB.center
        torque_B = np.mean(np.cross(levers_B, v_ext_at_B), axis=0)
        
        # Apply mutual Lagrangian feedback update
        pA.center += drift_A * dt
        pA.omega  += torque_A * dt
        pA.update_basis()
        
        pB.center += drift_B * dt
        pB.omega  += torque_B * dt
        pB.update_basis()
        
    return hist_pos_A, hist_pos_B, hist_n_A, hist_n_B

# --- EXECUTE SANDBOX RUN ---
if __name__ == "__main__":
    # Particle A: Stationary heavy proton anchor centered near origin
    core_A = TrackedToroid(
        center=[0.0, 0.0, 0.0],
        major_radius=0.5,
        minor_radius=0.1,
        vorticity_vec=[0.0, 0.0, 25.0] # High vertical torque spinning around Z
    )
    
    # Particle B: Lighter orbital test body offset along X-axis, tilted intentionally
    core_B = TrackedToroid(
        center=[1.8, 0.0, 0.0],
        major_radius=0.4,
        minor_radius=0.1,
        vorticity_vec=[8.0, 8.0, 0.0] # Tilted into the X-Y plane to trigger precession
    )
    
    # Run the interactive clock loop
    steps_total = 500
    dt_delta = 0.02
    posA, posB, normA, normB = run_advection_loop(core_A, core_B, steps=steps_total, dt=dt_delta)
    
    print("=== TIR LONG-TERM TIMELINE SIMULATION COMPLETE ===")
    print(f"Final Position core_A: {posA[-1]}")
    print(f"Final Position core_B: {posB[-1]}")

    # --- MATPLOTLIB DUAL-PANEL RENDERING ---
    fig = plt.figure(figsize=(15, 7), facecolor='#111116')
    
    # PANEL 1: 3D Spatial Trajectories
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor('#111116')
    ax1.w_xaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    ax1.w_yaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    ax1.w_zaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    
    # Plot spatial tracks
    ax1.plot(posA[:,0], posA[:,1], posA[:,2], color='magenta', linewidth=2.0, label='Core A (Proton Anchor)')
    ax1.plot(posB[:,0], posB[:,1], posB[:,2], color='cyan', linewidth=2.0, label='Core B (Orbital Test Core)')
    # Final location markers
    ax1.scatter(posA[-1,0], posA[-1,1], posA[-1,2], color='magenta', s=40)
    ax1.scatter(posB[-1,0], posB[-1,1], posB[-1,2], color='cyan', s=40)
    
    ax1.set_title("Emergent Substrate Trajectories (Spatial Orbits)", color='white', fontsize=12, pad=10)
    ax1.set_xlabel("X Space", color='white')
    ax1.set_ylabel("Y Space", color='white')
    ax1.set_zlabel("Z Space", color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#1c1c24', edgecolor='none', labelcolor='white')
    
    # PANEL 2: Precessional Cone Spherical Mapping
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor('#111116')
    ax2.xaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    ax2.yaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    ax2.zaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    
    # Draw reference wireframe sphere to help contextualize orientation tilt
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    ax2.plot_wireframe(xs, ys, zs, color='#222233', linewidth=0.5, alpha=0.4)
    
    # Plot spin orientation vector evolution over time
    ax2.plot(normA[:,0], normA[:,1], normA[:,2], color='magenta', linewidth=1.5, label='Axis A Track')
    ax2.plot(normB[:,0], normB[:,1], normB[:,2], color='cyan', linewidth=2.5, label='Axis B Precession Track')
    
    # Start and end markers for orientation axis
    ax2.scatter(normB[0,0], normB[0,1], normB[0,2], color='yellow', s=50, marker='*', label='Axis B Setup Start')
    ax2.scatter(normB[-1,0], normB[-1,1], normB[-1,2], color='cyan', s=30, marker='o')
    
    ax2.set_title("Spin Orientation Vector Track (Gyroscopic Precession)", color='white', fontsize=12, pad=10)
    ax2.set_xlabel("Normal Nx", color='white')
    ax2.set_ylabel("Normal Ny", color='white')
    ax2.set_zlabel("Normal Nz", color='white')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#1c1c24', edgecolor='none', labelcolor='white')
    
    # Fixed coordinate lock for axis bounding representation
    ax2.set_xlim([-1, 1]); ax2.set_ylim([-1, 1]); ax2.set_zlim([-1, 1])


    
    plt.tight_layout()
    plt.show()