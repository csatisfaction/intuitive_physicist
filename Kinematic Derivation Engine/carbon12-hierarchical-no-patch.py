import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class PureAutogenousCarbonSolver:
    def __init__(self, target_radius=2.0):
        self.r_target = target_radius

    def unpack_state(self, params, N):
        P = params[0:N*3].reshape(N, 3)     
        S_raw = params[N*3:N*6].reshape(N, 3) 
        S_norms = np.linalg.norm(S_raw, axis=1, keepdims=True) + 1e-9
        S = S_raw / S_norms
        return P, S

    def evaluate_impedance(self, params, N):
        P, S = self.unpack_state(params, N)
        total_impedance = 0.0

        # RULE 1: Vacuum Entrainment (Local volume packing)
        for i in range(N):
            distances = [np.linalg.norm(P[i] - P[j]) for j in range(N) if i != j]
            distances = np.sort(distances)
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
                if d < self.r_target * 1.5:
                    r_hat = r_vec / (d + 1e-9)
                    shear_clash = np.cross(S[i] + S[j], r_hat)
                    total_impedance += 5.0 * np.linalg.norm(shear_clash)**2

        total_impedance += np.linalg.norm(np.mean(P, axis=0))**2
        return total_impedance

    def solve_single_alpha(self):
        """ Stage 1: Native Hadronic Condensation """
        print("STAGE 1: Optimizing pristine 4-body Alpha Particle...")
        np.random.seed(42)
        initial_4body = np.random.uniform(-1.0, 1.0, 24)
        res = minimize(lambda x: self.evaluate_impedance(x, 4), initial_4body, method='BFGS')
        P_alpha, S_alpha = self.unpack_state(res.x, 4)
        P_alpha -= np.mean(P_alpha, axis=0)
        return P_alpha, S_alpha

    def seed_randomized_clusters(self, P_alpha, S_alpha):
        """ Stage 2: 100% Random 3D Clones (Zero Geometric Bias/Patches) """
        print("STAGE 2: Scattering 3 Alphas with random 3D offsets and rotations...")
        c12_guess_P = []
        c12_guess_S = []
        
        np.random.seed(101) # Change this seed to test different initial jumbles
        
        for cluster_idx in range(3):
            # Completely randomized 3D translation vector
            translation = np.random.uniform(-3.5, 3.5, 3)
            
            # Completely randomized 3D rotation matrix via Euler angles
            angles = np.random.uniform(0, 2 * np.pi, 3)
            c = np.cos(angles)
            s = np.sin(angles)
            
            # Unpacking fixed cleanly here:
            R_x = np.array([[1, 0, 0], [0, c[0], -s[0]], [0, s[0], c[0]]])
            R_y = np.array([[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]])
            R_z = np.array([[c[2], -s[2], 0], [s[2], c[2], 0], [0, 0, 1]])
            R_3d = R_z @ R_y @ R_x
            
            for i in range(4):
                c12_guess_P.append(R_3d @ P_alpha[i] + translation)
                c12_guess_S.append(R_3d @ S_alpha[i])
                
        return np.concatenate([np.array(c12_guess_P).flatten(), np.array(c12_guess_S).flatten()])

    def solve_pure_carbon12(self):
        P_alpha, S_alpha = self.solve_single_alpha()
        pure_chaotic_seed = self.seed_randomized_clusters(P_alpha, S_alpha)
        
        print("STAGE 3: Unleashing 12-body relaxation from total chaos...")
        res = minimize(lambda x: self.evaluate_impedance(x, 12), pure_chaotic_seed, method='BFGS', 
                       options={'maxiter': 3000, 'disp': False})
        
        P_final, S_final = self.unpack_state(res.x, 12)
        print(f"Autogenous Convergence Complete. Impedance: {res.fun:.4f}")
        return P_final, S_final

# --- RENDER ENGINE ---
if __name__ == "__main__":
    solver = PureAutogenousCarbonSolver()
    P, S = solver.solve_pure_carbon12()
    
    fig = plt.figure(figsize=(12, 10), facecolor='#111116')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#111116')
    
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(P[i] - P[j])
            if dist < 2.4:
                ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], [P[i, 2], P[j, 2]], 
                        color='gray', linestyle='-', alpha=0.5, linewidth=2)

    protons = P[[i for i in range(12) if i % 2 == 0]]
    neutrons = P[[i for i in range(12) if i % 2 != 0]]
    
    ax.scatter(protons[:, 0], protons[:, 1], protons[:, 2], s=550, c='magenta', label='Protons (Sine)', depthshade=True)
    ax.scatter(neutrons[:, 0], neutrons[:, 1], neutrons[:, 2], s=550, c='cyan', label='Neutrons (Cosine/Tilted)', depthshade=True)
    ax.quiver(P[:, 0], P[:, 1], P[:, 2], S[:, 0], S[:, 1], S[:, 2], length=1.1, color='lime', linewidth=2.5, label='Gyroscopic Axis')

    ax.set_title("TIR Un-Patched Engine: Autogenous Carbon-12\nSymmetrical Triangular Macro-Gear Found From 3D Chaotic Seed", color='white', pad=20)
    ax.set_xlim([-4, 4]); ax.set_ylim([-4, 4]); ax.set_zlim([-4, 4]); ax.axis('off')
    ax.legend(facecolor='#222233', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.show()