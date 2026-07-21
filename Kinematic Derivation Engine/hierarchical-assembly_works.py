import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class PristineHierarchicalSolver:
    def __init__(self, core_diameter=2.0):
        self.r_target = core_diameter

    def unpack_state(self, params, N):
        P = params[0:N*3].reshape(N, 3)     
        S_raw = params[N*3:N*6].reshape(N, 3) 
        S_norms = np.linalg.norm(S_raw, axis=1, keepdims=True) + 1e-9
        S = S_raw / S_norms
        return P, S

    def pristine_pairwise_pressure(self, P_i, P_j, S_i, S_j):
        """
        The Clean, Un-Patched Kinematic Pressure Equation.
        P = ρ * (1 - α) + V_pull
        """
        r_vec = P_i - P_j
        d = np.linalg.norm(r_vec) + 1e-9
        u_hat = r_vec / d
        
        # 1. Map 3D fluid streaming vectors projected into the interface gap
        v_i = np.cross(S_i, u_hat)
        v_j = -np.cross(S_j, u_hat)
        
        norm_v_i = np.linalg.norm(v_i)
        norm_v_j = np.linalg.norm(v_j)
        
        # Derive pure vector alignment (α)
        alpha = np.dot(v_i, v_j) / (norm_v_i * norm_v_j + 1e-9) if (norm_v_i > 1e-5 and norm_v_j > 1e-5) else 0.0
        
        # 2. Isotropic Substrate Density Floor (Choked Flow Backlog)
        gap = d - self.r_target
        rho_backlog = 1.0 + 1.5 / (max(0.01, gap) ** 2)
        
        # The Pristine Kinetic Wall
        P_gap = rho_backlog * (1.0 - alpha)
        
        # Absolute Hard Core Exclusion: A node's solid physical volume cannot be zeroed out by spin alignment
        if gap < 0:
            P_gap += 5000.0 * (abs(gap) ** 3)
            
        # 3. The Scalar Acceleration Well (Long-range Entrainment Pull)
        V_pull = - 16.0 / (d + 0.1)
        
        return P_gap + V_pull

    def evaluate_alpha_impedance(self, params):
        """ Optimizes 4 nucleons into a stable Alpha tetrahedron using pure kinematics """
        P, S = self.unpack_state(params, 4)
        impedance = 0.0
        for i in range(4):
            for j in range(i+1, 4):
                impedance += self.pristine_pairwise_pressure(P[i], P[j], S[i], S[j])
        impedance += np.linalg.norm(np.mean(P, axis=0))**2 # Coordinate Anchor
        return impedance

    def build_locked_alpha_template(self):
        """ STEP 1: Relax a single Alpha cluster into a perfect, invariant gear-mesh unit """
        print("STEP 1: Relaxing pristine 4-body Alpha Tetrahedron template...")
        np.random.seed(101)
        initial_guess = np.random.uniform(-1.0, 1.0, 24)
        res = minimize(self.evaluate_alpha_impedance, initial_guess, method='BFGS')
        P_alpha, S_alpha = self.unpack_state(res.x, 4)
        P_alpha -= np.mean(P_alpha, axis=0) # Lock to local origin
        return P_alpha, S_alpha

    def evaluate_carbon_macro_impedance(self, macro_params, P_template, S_template):
        """
        STEP 2: Optimize the macro-ring using locked Alpha clusters as rigid building blocks.
        macro_params contains: [radius_ring, theta_rotation_A, theta_rotation_B, theta_rotation_C]
        """
        ring_R = macro_params[0]
        thetas = macro_params[1:4]
        
        # Reconstruct the 12-body positions based on the macro-parameters
        P_global = []
        S_global = []
        
        for idx in range(3):
            angle = idx * (2 * np.pi / 3)
            # Translate cluster center out to the macro-ring horizon
            center_pos = np.array([ring_R * np.cos(angle), ring_R * np.sin(angle), 0.0])
            
            # Rotate the entire cluster to test alternative orbital gear-meshing configurations
            phi = thetas[idx]
            R_z = np.array([
                [np.cos(phi), -np.sin(phi), 0.0],
                [np.sin(phi),  np.cos(phi), 0.0],
                [0.0,          0.0,         1.0]
            ])
            
            for i in range(4):
                P_global.append(R_z @ P_template[i] + center_pos)
                S_global.append(R_z @ S_template[i])
                
        P_global = np.array(P_global)
        S_global = np.array(S_global)
        
        # Tally total system impedance across all 12 nodes using the pristine equation
        total_impedance = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                total_impedance += self.pristine_pairwise_pressure(P_global[i], P_global[j], S_global[i], S_global[j])
                
        return total_impedance

    def assemble_carbon12(self):
        # Step 1: Secure the locked, stable building block of the lower octave
        P_alpha, S_alpha = self.build_locked_alpha_template()
        
        # Step 2: Define macro-envelope optimization parameters [Ring Radius, Rotations of the 3 Alphas]
        print("STEP 2: Assembling Carbon-12 macro-envelope via rigid Alpha gear-meshing...")
        initial_macro_guess = np.array([self.r_target * 1.3, 0.0, 0.0, 0.0])
        
        res = minimize(lambda x: self.evaluate_carbon_macro_impedance(x, P_alpha, S_alpha), 
                       initial_macro_guess, method='Nelder-Mead')
        
        # Reconstruct final system coordinates from the winning macro-configuration
        final_R = res.x[0]
        final_thetas = res.x[1:4]
        
        P_final, S_final = [], []
        for idx in range(3):
            angle = idx * (2 * np.pi / 3)
            center_pos = np.array([final_R * np.cos(angle), final_R * np.sin(angle), 0.0])
            phi = final_thetas[idx]
            R_z = np.array([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]])
            
            for i in range(4):
                P_final.append(R_z @ P_alpha[i] + center_pos)
                S_final.append(R_z @ S_alpha[i])
                
        print(f"Hierarchical Assembly Complete. Inter-Cluster Gear Mesh Achieved.")
        return np.array(P_final), np.array(S_final)

# --- EXECUTE AND RENDER TRUE HIERARCHICAL OCTAVE ---
if __name__ == "__main__":
    solver = PristineHierarchicalSolver()
    P, S = solver.assemble_carbon12()
    
    fig = plt.figure(figsize=(12, 10), facecolor='#111116')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#111116')
    
    # Render geometric structural linkages between touching nucleon boundaries
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(P[i] - P[j])
            if dist < 2.5: 
                ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], [P[i, 2], P[j, 2]], 
                        color='white', linestyle='-', alpha=0.4, linewidth=2.0)

    # Differentiate alternating structural layers
    protons = P[[i for i in range(12) if i % 2 == 0]]
    neutrons = P[[i for i in range(12) if i % 2 != 0]]
    
    ax.scatter(protons[:, 0], protons[:, 1], protons[:, 2], s=700, c='#ff00ff', label='Proton Nodes', depthshade=True)
    ax.scatter(neutrons[:, 0], neutrons[:, 1], neutrons[:, 2], s=700, c='#00ffff', label='Neutron Nodes', depthshade=True)

    # High-contrast render of the interlocking 3D spin gear axes
    ax.quiver(P[:, 0], P[:, 1], P[:, 2], S[:, 0], S[:, 1], S[:, 2], 
              length=0.7, color='#00ff00', arrow_length_ratio=0.2, linewidth=2.5, label=r'Co-Shear Spin Axis ($\vec{S}$)')

    ax.set_title("Carbon-12 ($^{12}C$) Fractal Stereochemistry\nBottom-Up Hierarchical Assembly via Locked Alpha Sub-Units", color='white', pad=20, fontsize=13, fontweight='bold')
    ax.set_xlim([-4, 4]); ax.set_ylim([-4, 4]); ax.set_zlim([-4, 4])
    ax.axis('off')
    ax.legend(facecolor='#222233', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.show()