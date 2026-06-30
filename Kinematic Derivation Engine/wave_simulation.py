import numpy as np
import matplotlib.pyplot as plt

class PureWaveToroid:
    def __init__(self, center, major_radius, minor_radius, vorticity_vec, pulse_amp=0.3, freq=20.0, phase=0.0, c=1.0):
        """
        First-Principles Toroidal Soliton Core emitting active AC/DC substrate waves.
        """
        self.center = np.array(center, dtype=np.float64)
        self.R = float(major_radius)
        self.r_m = float(minor_radius)
        self.omega = np.array(vorticity_vec, dtype=np.float64)
        self.c = float(c)
        
        # AC Wave Parameters (Radial Breathing Primitives)
        self.pulse_amp = float(pulse_amp)
        self.freq = float(freq)
        self.phase = float(phase)
        
        self.omega_mag = np.linalg.norm(self.omega)
        self.circulation = self.omega_mag * np.pi * (self.r_m ** 2)
        self.update_basis()

    def update_basis(self):
        self.omega_mag = np.linalg.norm(self.omega)
        if self.omega_mag > 1e-12:
            self.normal = self.omega / self.omega_mag
        else:
            self.normal = np.array([0.0, 0.0, 1.0])
            
        arbitrary_vec = np.array([0.0, 0.0, 1.0]) if abs(self.normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        self.u_basis = np.cross(self.normal, arbitrary_vec)
        self.u_basis /= np.linalg.norm(self.u_basis)
        self.v_basis = np.cross(self.normal, self.u_basis)

    def get_filament_nodes(self, num_segments=32):
        theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        cos_t = np.cos(theta)[:, np.newaxis]
        sin_t = np.sin(theta)[:, np.newaxis]
        return self.center + self.R * (cos_t * self.u_basis + sin_t * self.v_basis)

    def project_field(self, target_point, t, num_segments=32):
        """
        Projects superimposed steady circulation and radiating phase-retarded waves
        into any point in space, bounded strictly by velocity saturation ceiling (c).
        """
        x = np.array(target_point, dtype=np.float64)
        v_raw = np.zeros(3, dtype=np.float64)
        ring_nodes = self.get_filament_nodes(num_segments)
        
        for i in range(num_segments):
            node_current = ring_nodes[i]
            node_next = ring_nodes[(i + 1) % num_segments]
            
            delta_s = node_next - node_current
            midpoint = (node_current + node_next) * 0.5
            r_vec = x - midpoint
            r_mag = np.linalg.norm(r_vec)
            
            if r_mag < 1e-12:
                continue
                
            # 1. Steady-state DC Vortex Circulation (Biot-Savart)
            r_mag_sq = r_mag ** 2
            denominator_dc = (r_mag_sq + (self.r_m ** 2)) ** 1.5
            cross_prod = np.cross(delta_s, r_vec)
            v_dc = (self.circulation / (4 * np.pi)) * (cross_prod / denominator_dc)
            
            # 2. Retarded AC Phase Wave Emission (Radial Breathing Pulsation)
            # Incorporates spatial time-of-flight phase lag (distance / c)
            time_of_flight_lag = r_mag / self.c
            wave_term = np.cos(self.freq * (t - time_of_flight_lag) + self.phase)
            v_ac = self.pulse_amp * wave_term * (r_vec / r_mag)
            
            v_raw += v_dc + v_ac
            
        # 3. CRITICAL UNYIELDING SATURATION CEILING CONSTRAINT (v <= c)
        v_mag = np.linalg.norm(v_raw)
        if v_mag > self.c:
            return self.c * (v_raw / v_mag)
            
        return v_raw

def run_pure_wave_interaction(pA, pB, steps=300, dt=0.02):
    """
    Executes a pure, un-patched Lagrangian interaction timeline.
    Cores update entirely based on the local background substrate fields.
    """
    hist_pos_A = np.zeros((steps, 3))
    hist_pos_B = np.zeros((steps, 3))
    hist_n_B   = np.zeros((steps, 3))
    
    for step in range(steps):
        t = step * dt
        hist_pos_A[step] = pA.center.copy()
        hist_pos_B[step] = pB.center.copy()
        hist_n_B[step]   = pB.normal.copy()
        
        # Particle A reads Particle B's active field at time t
        nodes_A = pA.get_filament_nodes()
        v_ext_at_A = np.array([pB.project_field(n, t) for n in nodes_A])
        drift_A = np.mean(v_ext_at_A, axis=0)
        torque_A = np.mean(np.cross(nodes_A - pA.center, v_ext_at_A), axis=0)
        
        # Particle B reads Particle A's active field at time t
        nodes_B = pB.get_filament_nodes()
        v_ext_at_B = np.array([pA.project_field(n, t) for n in nodes_B])
        drift_B = np.mean(v_ext_at_B, axis=0)
        torque_B = np.mean(np.cross(nodes_B - pB.center, v_ext_at_B), axis=0)
        
        # Pure Advection Update: The center moves directly with the background transport current
        pA.center += drift_A * dt
        pA.omega  += torque_A * dt
        pA.update_basis()
        
        pB.center += drift_B * dt
        pB.omega  += torque_B * dt
        pB.update_basis()
        
    return hist_pos_A, hist_pos_B, hist_n_B

if __name__ == "__main__":
    # Particle A: High-torque stationary proton anchor pulsing at origin
    proton = PureWaveToroid(
        center=[0.0, 0.0, 0.0], major_radius=0.5, minor_radius=0.1, 
        vorticity_vec=[0.0, 0.0, 30.0], pulse_amp=0.4, freq=25.0, phase=0.0
    )
    
    # Particle B: Neighboring core offset along X-axis
    # Out of phase (phase=pi) relative to Particle A to observe structural attraction/repulsion dynamics
    test_core = PureWaveToroid(
        center=[1.7, 0.0, 0.0], major_radius=0.4, minor_radius=0.1, 
        vorticity_vec=[6.0, 6.0, 0.0], pulse_amp=0.4, freq=25.0, phase=np.pi
    )
    
    steps_total = 400
    dt_delta = 0.02
    posA, posB, normB = run_pure_wave_interaction(proton, test_core, steps=steps_total, dt=dt_delta)
    
    print("=== PURE WAVE FIRST-PRINCIPLES SIMULATION COMPLETE ===")
    
    # --- PLOTTING OUTPUTS ---
    fig = plt.figure(figsize=(14, 6), facecolor='#111116')
    
    # Left Panel: True First-Principles Spatial Trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor('#111116')
    ax1.xaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    ax1.yaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    ax1.zaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    
    ax1.plot(posA[:,0], posA[:,1], posA[:,2], color='magenta', linewidth=2.0, label='Core A (Proton)')
    ax1.plot(posB[:,0], posB[:,1], posB[:,2], color='cyan', linewidth=2.0, label='Core B (Orbital Core)')
    ax1.scatter(posA[-1,0], posA[-1,1], posA[-1,2], color='magenta', s=40)
    ax1.scatter(posB[-1,0], posB[-1,1], posB[-1,2], color='cyan', s=40)
    
    ax1.set_title("Pure Wave Emergent Substrate Trajectories", color='white', fontsize=12, pad=10)
    ax1.set_xlabel("X Space", color='white'); ax1.set_ylabel("Y Space", color='white'); ax1.set_zlabel("Z Space", color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#1c1c24', edgecolor='none', labelcolor='white')
    
    # Right Panel: Gyroscopic Precession Track
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor('#111116')
    ax2.xaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    ax2.yaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    ax2.zaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    
    u_g, v_g = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xs = np.cos(u_g) * np.sin(v_g)
    ys = np.sin(u_g) * np.sin(v_g)
    zs = np.cos(v_g)
    ax2.plot_wireframe(xs, ys, zs, color='#222233', linewidth=0.5, alpha=0.4)
    
    ax2.plot(normB[:,0], normB[:,1], normB[:,2], color=(0.0, 1.0, 1.0, 0.8), linewidth=2.5, label='Axis B Precession')
    ax2.scatter(normB[0,0], normB[0,1], normB[0,2], color='yellow', s=60, marker='*', label='Precession Start')
    
    ax2.set_title("Pure Wave Spin Axis Precession Track", color='white', fontsize=12, pad=10)
    ax2.set_xlabel("Nx", color='white'); ax2.set_ylabel("Ny", color='white'); ax2.set_zlabel("Nz", color='white')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#1c1c24', edgecolor='none', labelcolor='white')
    ax2.set_xlim([-1, 1]); ax2.set_ylim([-1, 1]); ax2.set_zlim([-1, 1])
    
    plt.tight_layout()
    plt.show()