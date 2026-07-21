import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class PureKinematicNuclearSolver:
    def __init__(self, core_diameter=2.0):
        self.r_target = core_diameter  

    def unpack_state(self, params, N):
        P = params[0:N*3].reshape(N, 3)     
        S_raw = params[N*3:N*6].reshape(N, 3) 
        S_norms = np.linalg.norm(S_raw, axis=1, keepdims=True) + 1e-9
        S = S_raw / S_norms
        return P, S

    def evaluate_impedance(self, params, N):
        """
        Unified Kinematic Pressure Impedance Function (Loophole Fixed).
        Ensures isotropic compression remains active even during perfect alignment.
        """
        P, S = self.unpack_state(params, N)
        total_impedance = 0.0

        for i in range(N):
            for j in range(i+1, N):
                r_vec = P[i] - P[j]
                d = np.linalg.norm(r_vec) + 1e-9
                u_hat = r_vec / d
                
                # 1. 3D Gap Interface Velocity Mapping
                v_i = np.cross(S[i], u_hat)
                v_j = -np.cross(S[j], u_hat) 
                
                norm_v_i = np.linalg.norm(v_i)
                norm_v_j = np.linalg.norm(v_j)
                
                if norm_v_i > 1e-5 and norm_v_j > 1e-5:
                    alpha = np.dot(v_i, v_j) / (norm_v_i * norm_v_j)
                else:
                    alpha = 0.0 
                
                # 2. The Stop - Isotropic Compression + Alignment Modulation
                gap = d - self.r_target
                
                if gap > 0:
                    # Continuous asymptotic wall as gap closes
                    rho_backlog = 1.0 + 2.5 / (gap ** 2)
                else:
                    # Severe polynomial penalty if the optimizer attempts to force an overlap
                    rho_backlog = 1.0 + 2.5 / (0.01**2) + 5000.0 * (abs(gap) ** 3)
                
                # LOOPHOLE FIX: (2.0 - alpha) ensures the wall stays active even if alpha == 1
                P_gap = rho_backlog * (2.0 - alpha)
                
                # 3. The Pull - Substrate Acceleration Well
                V_pull = - 18.0 / (d + 0.1)
                
                total_impedance += P_gap + V_pull

        # System Anchor
        total_impedance += np.linalg.norm(np.mean(P, axis=0))**2
        return total_impedance

    def solve_single_alpha(self):
        print("STAGE 1: Relaxing pristine 4-body Alpha Tetrahedron...")
        np.random.seed(42)
        initial_4body = np.random.uniform(-1.0, 1.0, 24)
        
        res = minimize(lambda x: self.evaluate_impedance(x, 4), initial_4body, method='BFGS')
        P_alpha, S_alpha = self.unpack_state(res.x, 4)
        P_alpha -= np.mean(P_alpha, axis=0) 
        return P_alpha, S_alpha

    def assemble_carbon_ring(self, P_alpha, S_alpha):
        print("STAGE 2: Arranging Alpha templates into triangular macro-envelope...")
        c12_guess_P = []
        c12_guess_S = []
        ring_radius = self.r_target * 1.35 
        
        for cluster_idx in range(3):
            angle = cluster_idx * (2 * np.pi / 3)
            translation = np.array([ring_radius * np.cos(angle), ring_radius * np.sin(angle), 0.0])
            
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R_z = np.array([
                [cos_a, -sin_a, 0.0],
                [sin_a,  cos_a, 0.0],
                [0.0,    0.0,   1.0]
            ])
            
            for i in range(4):
                c12_guess_P.append(R_z @ P_alpha[i] + translation)
                c12_guess_S.append(R_z @ S_alpha[i])
                
        return np.concatenate([np.array(c12_guess_P).flatten(), np.array(c12_guess_S).flatten()])

    def solve_carbon12(self):
        P_alpha_template, S_alpha_template = self.solve_single_alpha()
        structured_guess = self.assemble_carbon_ring(P_alpha_template, S_alpha_template)
        
        print("STAGE 3: Launching global 12-body un-patched substrate relaxation...")
        res = minimize(lambda x: self.evaluate_impedance(x, 12), structured_guess, method='BFGS', 
                       options={'maxiter': 3500, 'disp': False})
        
        P_final, S_final = self.unpack_state(res.x, 12)
        print(f"Kinematic Convergence Complete. Final System Impedance Score: {res.fun:.4f}")
        return P_final, S_final

if __name__ == "__main__":
    solver = PureKinematicNuclearSolver()
    P, S = solver.solve_carbon12()
    
    fig = plt.figure(figsize=(12, 10), facecolor='#111116')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#111116')
    
    # Draw links between valid touching interfaces
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(P[i] - P[j])
            if dist < 2.5: 
                ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], [P[i, 2], P[j, 2]], 
                        color='white', linestyle='-', alpha=0.3, linewidth=1.5)

    protons = P[[i for i in range(12) if i % 2 == 0]]
    neutrons = P[[i for i in range(12) if i % 2 != 0]]
    
    ax.scatter(protons[:, 0], protons[:, 1], protons[:, 2], s=600, c='#ff00ff', label='Proton Nodes', depthshade=True)
    ax.scatter(neutrons[:, 0], neutrons[:, 1], neutrons[:, 2], s=600, c='#00ffff', label='Neutron Nodes', depthshade=True)

    # Added raw string 'r' prefix to completely eliminate the Matplotlib LaTeX crash
    ax.quiver(P[:, 0], P[:, 1], P[:, 2], S[:, 0], S[:, 1], S[:, 2], 
              length=0.8, color='#00ff00', arrow_length_ratio=0.2, linewidth=2, label=r'Co-Shear Spin Axis ($\vec{S}$)')

    ax.set_title("Carbon-12 ($^{12}C$) Stereochemistry\nEmergent Structural Relaxation via Pure Kinematic Pressure Field", color='white', pad=20, fontsize=13, fontweight='bold')
    ax.set_xlim([-4, 4]); ax.set_ylim([-4, 4]); ax.set_zlim([-4, 4])
    ax.axis('off')
    ax.legend(facecolor='#222233', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.show()