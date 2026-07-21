import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class HierarchicalNuclearSolver:
    def __init__(self, target_radius=2.0):
        self.r_target = target_radius

    def unpack_state(self, params, N):
        P = params[0:N*3].reshape(N, 3)     
        S_raw = params[N*3:N*6].reshape(N, 3) 
        S_norms = np.linalg.norm(S_raw, axis=1, keepdims=True) + 1e-9
        S = S_raw / S_norms
        return P, S

    def evaluate_impedance(self, params, N):
        """ Pristine TIR fluid impedance function used for both Alpha and Carbon stages """
        P, S = self.unpack_state(params, N)
        total_impedance = 0.0

        # RULE 1: Vacuum Entrainment (Volume Packing)
        for i in range(N):
            distances = [np.linalg.norm(P[i] - P[j]) for j in range(N) if i != j]
            distances = np.sort(distances)
            # Tetrahedral packing means each nucleon locks with its 3 closest neighbors
            for d in distances[:3]: 
                total_impedance += 15.0 * (d - self.r_target)**2

        # RULE 2: Coulomb Repulsion (Protons = Even Indices)
        protons = [i for i in range(N) if i % 2 == 0]
        for i in range(len(protons)):
            for j in range(i+1, len(protons)):
                d_protons = np.linalg.norm(P[protons[i]] - P[protons[j]])
                total_impedance += 6.0 / (d_protons + 0.1)

        # RULE 3: Kinematic Co-Shear (Tangential Gear Meshing)
        for i in range(N):
            for j in range(i+1, N):
                r_vec = P[i] - P[j]
                d = np.linalg.norm(r_vec)
                if d < self.r_target * 1.5: # Boundaries must be touching to shear
                    r_hat = r_vec / (d + 1e-9)
                    shear_clash = np.cross(S[i] + S[j], r_hat)
                    total_impedance += 5.0 * np.linalg.norm(shear_clash)**2

        # System Anchor: Prevents the molecule from drifting out of view
        total_impedance += np.linalg.norm(np.mean(P, axis=0))**2
        return total_impedance

    def solve_single_alpha(self):
        """ Stage 1: Generate the perfect first-principles Alpha Template """
        print("STAGE 1: Optimizing pristine 4-body Alpha Particle...")
        np.random.seed(42)
        initial_4body = np.random.uniform(-1.0, 1.0, 24)
        
        res = minimize(lambda x: self.evaluate_impedance(x, 4), initial_4body, method='BFGS')
        P_alpha, S_alpha = self.unpack_state(res.x, 4)
        
        # Center the alpha template at local coordinate zero
        P_alpha -= np.mean(P_alpha, axis=0)
        return P_alpha, S_alpha

    def assemble_carbon_ring(self, P_alpha, S_alpha):
        """ Stage 2 & 3: Clone, rotate, and arrange the Alphas into a triangular layout """
        print("STAGE 2: Replicating Alpha coordinates into triangular macro-layout...")
        c12_guess_P = []
        c12_guess_S = []
        
        # Radius of the macro-triangle ring (calculated to prevent initial cluster smashing)
        ring_radius = self.r_target * 1.2
        
        for cluster_idx in range(3):
            angle = cluster_idx * (2 * np.pi / 3)
            
            # Spatial translation vector for this cluster center
            translation = np.array([ring_radius * np.cos(angle), ring_radius * np.sin(angle), 0.0])
            
            # Simple 2D rotation matrix around Z to keep cluster faces oriented relational to center
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R_z = np.array([
                [cos_a, -sin_a, 0.0],
                [sin_a,  cos_a, 0.0],
                [0.0,    0.0,   1.0]
            ])
            
            # Transform and append the template nucleon variables
            for i in range(4):
                rotated_pos = R_z @ P_alpha[i]
                rotated_spin = R_z @ S_alpha[i]
                
                c12_guess_P.append(rotated_pos + translation)
                c12_guess_S.append(rotated_spin)
                
        # Flatten into the 72-variable format required by SciPy
        return np.concatenate([np.array(c12_guess_P).flatten(), np.array(c12_guess_S).flatten()])

    def solve_carbon12(self):
        # Step 1 & 2: Get the structured layout using the exact alpha coordinates
        P_alpha_template, S_alpha_template = self.solve_single_alpha()
        structured_guess = self.assemble_carbon_ring(P_alpha_template, S_alpha_template)
        
        # Step 3: Unleash the individual 12-body optimizer
        print("STAGE 3: Running global 12-body substrate relaxation...")
        res = minimize(lambda x: self.evaluate_impedance(x, 12), structured_guess, method='BFGS', 
                       options={'maxiter': 2500, 'disp': False})
        
        P_final, S_final = self.unpack_state(res.x, 12)
        print(f"Hierarchical Convergence Complete. Impedance: {res.fun:.4f}")
        return P_final, S_final

# --- RUN AND RENDER PRODUCTION LINE ---
if __name__ == "__main__":
    solver = HierarchicalNuclearSolver()
    P, S = solver.solve_carbon12()
    
    # 3D Visualizer Render Setup
    fig = plt.figure(figsize=(12, 10), facecolor='#111116')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#111116')
    
    # Draw local fluid links between touching boundaries
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(P[i] - P[j])
            if dist < 2.4: # True contact boundary limit
                # Typo corrected here: P[j, j] changed to P[j, 1]
                ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], [P[i, 2], P[j, 2]], 
                        color='gray', linestyle='-', alpha=0.5, linewidth=2)

    # Differentiate Protons (Even indices) and Neutrons (Odd indices)
    protons = P[[i for i in range(12) if i % 2 == 0]]
    neutrons = P[[i for i in range(12) if i % 2 != 0]]
    
    ax.scatter(protons[:, 0], protons[:, 1], protons[:, 2], s=550, c='magenta', label='Protons (Sine)', depthshade=True)
    ax.scatter(neutrons[:, 0], neutrons[:, 1], neutrons[:, 2], s=550, c='cyan', label='Neutrons (Cosine/Tilted)', depthshade=True)

    # Trace the gyroscopic circulation vectors
    ax.quiver(P[:, 0], P[:, 1], P[:, 2], S[:, 0], S[:, 1], S[:, 2], 
              length=1.1, color='lime', arrow_length_ratio=0.25, linewidth=2.5, label='Gyroscopic Axis ($v_\\theta$)')

    ax.set_title("TIR Hierarchical Engine: Carbon-12 ($^{12}C$)\nThree Structural Alpha Tetrahedrons Relaxed Edge-to-Edge", color='white', pad=20, fontsize=13)
    ax.set_xlim([-4, 4]); ax.set_ylim([-4, 4]); ax.set_zlim([-4, 4])
    ax.axis('off')
    ax.legend(facecolor='#222233', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.show()