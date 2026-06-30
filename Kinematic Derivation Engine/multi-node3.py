import numpy as np
import matplotlib.pyplot as plt

class StabilizedToroid:
    def __init__(self, center, radius, vorticity_vec, num_nodes=24, minor_radius=0.15, pulse_amp=0.3, freq=15.0, phase=0.0, c=1.0):
        """
        A first-principles toroidal core with a rigid centripetal anchor (a = v^2/r) 
        preventing loop collapse during steep gradient advection.
        """
        self.N = num_nodes
        self.r_m = float(minor_radius)
        self.c = float(c)
        self.pulse_amp = float(pulse_amp)
        self.freq = float(freq)
        self.phase = float(phase)
        
        self.target_radius = float(radius)
        center = np.array(center, dtype=np.float64)
        omega = np.array(vorticity_vec, dtype=np.float64)
        omega_mag = np.linalg.norm(omega)
        self.normal = omega / omega_mag if omega_mag > 1e-12 else np.array([0.0, 0.0, 1.0])
        
        self.circulation = omega_mag * np.pi * (self.r_m ** 2)
        
        # Build initial coordinate ring layout
        arb = np.array([0.0, 0.0, 1.0]) if abs(self.normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        self.u_basis = np.cross(self.normal, arb); self.u_basis /= np.linalg.norm(self.u_basis)
        self.v_basis = np.cross(self.normal, self.u_basis)
        
        theta = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
        self.nodes = np.zeros((self.N, 3), dtype=np.float64)
        for i in range(self.N):
            self.nodes[i] = center + self.target_radius * (np.cos(theta[i]) * self.u_basis + np.sin(theta[i]) * self.v_basis)
            
        self.target_spacing = 2 * self.target_radius * np.sin(np.pi / self.N)

    def project_field(self, target_point, t):
        x = np.array(target_point, dtype=np.float64)
        v_total = np.zeros(3, dtype=np.float64)
        
        for i in range(self.N):
            node_curr = self.nodes[i]
            node_next = self.nodes[(i + 1) % self.N]
            
            delta_s = node_next - node_curr
            midpoint = (node_curr + node_next) * 0.5
            r_vec = x - midpoint
            r_mag = np.linalg.norm(r_vec)
            
            if r_mag < 1e-10:
                continue
                
            # DC Vortex Circulation Field
            v_dc = (self.circulation / (4 * np.pi)) * (np.cross(delta_s, r_vec) / ((r_mag**2 + (self.r_m ** 2)) ** 1.5))
            
            # AC Retarded Radial Wave Field
            time_of_flight_lag = r_mag / self.c
            v_ac = self.pulse_amp * np.cos(self.freq * (t - time_of_flight_lag) + self.phase) * (r_vec / r_mag)
            
            v_total += v_dc + v_ac
            
        return v_total

    def step_kinematics(self, external_field_evaluator, t, dt, suction_stiffness=0.3, core_rigidity=15.0):
        """
        Advects nodes via external pressure gradients while maintaining a strict 
        centripetal radius envelope to simulate internal vortex stability.
        """
        new_nodes = np.zeros_like(self.nodes)
        com = np.mean(self.nodes, axis=0) # Instantaneous center of mass
        eps = 0.03
        
        for i in range(self.N):
            curr_node = self.nodes[i]
            
            # 1. Evaluate External Bernoulli Kinetic Suction (nabla v^2)
            grad_v_sq = np.zeros(3)
            for axis in range(3):
                nudge = np.zeros(3)
                nudge[axis] = eps
                v_plus = np.linalg.norm(external_field_evaluator(curr_node + nudge, t))
                v_minus = np.linalg.norm(external_field_evaluator(curr_node - nudge, t))
                grad_v_sq[axis] = (v_plus**2 - v_minus**2) / (2 * eps)
            v_suction = grad_v_sq * suction_stiffness
            
            # 2. Apply Centripetal Radial Anchor Constraint (The Invariant Fix)
            vec_to_center = com - curr_node
            current_r = np.linalg.norm(vec_to_center)
            if current_r > 1e-10:
                # Restoring force to maintain the target major radius profile
                v_centripetal = core_rigidity * (current_r - self.target_radius) * (vec_to_center / current_r)
            else:
                v_centripetal = np.zeros(3)
                
            # Combine translation suction with internal radial stability
            v_node_raw = v_suction + v_centripetal
            
            # Velocity ceiling clamp (v <= c)
            v_node_mag = np.linalg.norm(v_node_raw)
            if v_node_mag > self.c:
                v_node_raw = self.c * (v_node_raw / v_node_mag)
                
            new_nodes[i] = curr_node + v_node_raw * dt
            
        self.nodes = new_nodes

    def get_center_of_mass(self):
        return np.mean(self.nodes, axis=0)


# --- EXECUTE STABILIZED SIMULATION ENGINE ---
if __name__ == "__main__":
    # Particle A: Symmetrical vertical spin anchor centered at origin
    core_A = StabilizedToroid(center=[0.0, 0.0, 0.0], radius=0.6, vorticity_vec=[0.0, 0.0, 35.0], phase=0.0)
    
    # Particle B: Stabilized mobile core offset along X axis, running out-of-phase
    core_B = StabilizedToroid(center=[2.3, 0.0, 0.0], radius=0.5, vorticity_vec=[0.0, 0.0, 25.0], phase=np.pi)
    
    steps = 140
    dt = 0.03
    
    com_A_track, com_B_track = [], []
    initial_nodes_B = core_B.nodes.copy()
    
    print("=== EXECUTING STABILIZED STRUCTURAL INTERACTION ===")
    for step in range(steps):
        t = step * dt
        com_A_track.append(core_A.get_center_of_mass())
        com_B_track.append(core_B.get_center_of_mass())
        
        # Symmetrical cross-probing evaluators
        eval_A = lambda pos, time: core_B.project_field(pos, time)
        eval_B = lambda pos, time: core_A.project_field(pos, time)
        
        core_A.step_kinematics(eval_A, t, dt)
        core_B.step_kinematics(eval_B, t, dt)
        
    com_A_track = np.array(com_A_track)
    com_B_track = np.array(com_B_track)
    
    # --- RENDER RESULTS ---
    plt.figure(figsize=(11, 6), facecolor='#111116')
    ax = plt.gca()
    ax.set_facecolor('#111116')
    
    plt.plot(core_A.nodes[:, 0], core_A.nodes[:, 1], color='magenta', alpha=0.6, label='Anchor Core A')
    plt.plot(initial_nodes_B[:, 0], initial_nodes_B[:, 1], color='cyan', linestyle='--', alpha=0.3, label='Core B Initial')
    
    # Plot final stabilized, structurally sound ring
    plt.plot(core_B.nodes[:, 0], core_B.nodes[:, 1], color='cyan', linewidth=2.5, label='Core B Final Stable Ring')
    plt.scatter(core_B.nodes[:, 0], core_B.nodes[:, 1], color='cyan', s=25)
    
    # Plot smooth Center of Mass migration path
    plt.plot(com_B_track[:, 0], com_B_track[:, 1], color='yellow', linestyle=':', linewidth=2.0, label='Emergent ATTRACTION Track')
    plt.scatter(com_B_track[0, 0], com_B_track[0, 1], color='yellow', marker='*', s=150, zorder=5)
    plt.scatter(com_B_track[-1, 0], com_B_track[-1, 1], color='lime', marker='D', s=60, label='COM Current', zorder=5)
    
    plt.title("TIR Stabilized Multi-Node Engine: Invariant Core Radius Attraction", color='white', fontsize=12)
    plt.xlabel("X Horizon", color='white'); plt.ylabel("Y Horizon", color='white')
    plt.grid(color='#222233', linestyle=':', linewidth=0.5)
    ax.tick_params(colors='white')
    ax.set_aspect('equal')
    plt.legend(facecolor='#1c1c24', edgecolor='none', labelcolor='white')
    plt.xlim([-1.0, 3.0]); plt.ylim([-1.5, 1.5])
    plt.show()